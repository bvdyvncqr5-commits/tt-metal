// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// WORKAROUND (compute_final_result is deliberately noinline): with fp32_dest_acc_en the intermediates
// are Float32, which enlarges this compute kernel enough that a fully-inlined kernel_main exceeds GCC's
// constant-folding budget. The LLK addrmod programmer emits SETC16 via an inline-asm "n" (immediate)
// operand whose whole instruction word — register address AND packed value — must fold to a compile-time
// constant (ckernel_addrmod.h); past the budget it can't, and JIT fails with "impossible constraint in
// 'asm'". Splitting the final-result loop into a noinline function shrinks kernel_main back under the
// budget so the immediates fold again. Behavior is unchanged (a plain function boundary); legacy compiled
// only because its smaller TU stayed under the budget. This is a stopgap: the proper fix is upstream in the
// LLK (make the addrmod value a compile-time constant, not just the index — templatizing the index alone
// was verified insufficient). See METAL2_PORT_REPORT.md.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Step 3 (final result) split out and forced noinline to keep kernel_main under the compiler's
// constant-folding budget — see the top-of-file WORKAROUND note. Body is verbatim from the inline loop.
static __attribute__((noinline)) void compute_final_result(
    DataflowBuffer& dfb_in0_obj,
    DataflowBuffer& dfb_max_obj,
    DataflowBuffer& dfb_tmp_obj,
    DataflowBuffer& dfb_exps_obj,
    DataflowBuffer& dfb_recipsumexps_obj,
    DataflowBuffer& dfb_out0_obj,
    std::uint32_t Wt) {
    constexpr std::uint32_t onetile = 1;
    for (std::uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
#ifdef SOFTMAX
        // x - max - log(sum)
        sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

        sub_tiles_bcast_cols_to_cb(dfb_tmp_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
        // -x + max - log(sum)
        // logsoftmin not implemented
#endif
#else
#ifdef SOFTMAX
        // exp(x - max) / sum
        sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

        exp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);

        mul_tiles_bcast_cols_to_cb(dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
        // rexp(x - max) / sum
        sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

        rexp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);

        mul_tiles_bcast_cols_to_cb(dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
    }
}

void kernel_main() {
    constexpr auto cb_in0 = dfb::in0;
    DataflowBuffer dfb_in0_obj(cb_in0);
    constexpr auto cb_mask = dfb::mask;
    DataflowBuffer dfb_mask_obj(cb_mask);
    constexpr auto cb_max_scaler = dfb::max_scaler;
    constexpr auto cb_sum_scaler = dfb::sum_scaler;
    constexpr auto cb_out0 = dfb::out0;
    DataflowBuffer dfb_out0_obj(cb_out0);
    constexpr auto cb_exps = dfb::exps;
    DataflowBuffer dfb_exps_obj(cb_exps);
    constexpr auto cb_recipsumexps = dfb::recip_sum_exps;
    DataflowBuffer dfb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = dfb::add;
    DataflowBuffer dfb_add_obj(cb_add);
    constexpr auto cb_max = dfb::max;
    DataflowBuffer dfb_max_obj(cb_max);
    constexpr auto cb_tmp = dfb::tmp;
    DataflowBuffer dfb_tmp_obj(cb_tmp);

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr std::uint32_t onetile = 1;

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing.
    std::uint32_t N = get_arg(args::N);
    std::uint32_t Wt = get_arg(args::Wt);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max
        if (Wt == 1) {
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into cb_max (no accumulation, first call).
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_in0, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile and continue reducing into cb_max via Accumulate.
            mask_tile_to_cb(dfb_in0_obj, dfb_mask_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, /*iter=*/1));
        }

        // step 1
        for (std::uint32_t w = 0; w < Wt; ++w) {
            // compute exp(x)
            if (w == Wt - 1) {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_and_mask_tile_to_cb(
                    dfb_tmp_obj,
                    dfb_mask_obj,
                    dfb_exps_obj,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#else
                rexp_tile_and_mask_tile_to_cb(
                    dfb_in0_obj,
                    dfb_mask_obj,
                    dfb_exps_obj,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
#endif
            } else {
#ifdef SOFTMAX
                sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

                exp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);
#else
                sub_tiles_bcast_cols_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

                rexp_tile_to_cb(dfb_tmp_obj, dfb_exps_obj);
#endif
            }

            if (w == 0) {
                copy_tile_to_cb(dfb_exps_obj, dfb_add_obj);
            } else {
                add_tiles_to_cb(dfb_add_obj, dfb_exps_obj, dfb_add_obj);
            }
        }

#ifdef LOG
        // compute log(sum) - pop tile after reduce
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_add,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::single(),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        // compute 1/sum(exp(x)) - pop tile after reduce
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_add,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::single(),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // step 3, compute final result (noinline — see top-of-file WORKAROUND note)
        compute_final_result(
            dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, dfb_exps_obj, dfb_recipsumexps_obj, dfb_out0_obj, Wt);

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
}
