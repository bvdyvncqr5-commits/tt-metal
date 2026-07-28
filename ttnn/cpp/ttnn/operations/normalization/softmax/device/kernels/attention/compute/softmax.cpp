// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "experimental/kernel_args.h"

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

template <std::uint32_t cb_in, std::uint32_t cb_max_scaler, std::uint32_t cb_max, std::uint32_t cb_out>
void calc_numeric_stable(std::uint32_t Wt, std::uint32_t ndst) {
    DataflowBuffer cb_in_obj(cb_in);
    DataflowBuffer cb_max_obj(cb_max);
    DataflowBuffer cb_out_obj(cb_out);

    // calculate max val per row
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        cb_in,
        cb_max_scaler,
        cb_max,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));

    // calculate x-max(x)
    exp_tile_init<EXP_APPROX>();
    reconfig_data_format_srcb(cb_max);
    cb_max_obj.wait_front(1);
    sub_bcast_cols_init_short(cb_in, cb_max);
    for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
        tile_regs_acquire();
        for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
            sub_tiles_bcast_cols(cb_in, cb_max, wt + wt8, 0, wt8);
        }
        cb_out_obj.reserve_back(ndst);
        for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
            exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
        }
        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
            pack_tile(wt8, cb_out);  // reuse the exps buffer again, this time in a circular manner
        }
        tile_regs_release();
        cb_out_obj.push_back(ndst);
    }
    cb_in_obj.pop_front(Wt);
    cb_max_obj.pop_front(1);
    cb_out_obj.wait_front(Wt);
}

void kernel_main() {
    const std::uint32_t NCHt = get_arg(args::num_rows);
    const std::uint32_t Ht = get_arg(args::Ht);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t ndst = get_arg(args::blk);
    const std::uint32_t start_ht = get_arg(args::start_ht);
    // The pad-mask data (W > W_unpadded) is host-known; it is carried as the MASK_PADDED_DATA compile-time
    // define (not a runtime arg), which lets the c_5 pad-mask DFB and the c_10 intermediate be bound only
    // on the paths that use them.

    constexpr std::uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_max_scaler = dfb::max_scaler;
    constexpr auto cb_sum_scaler = dfb::sum_scaler;
    constexpr auto cb_exps = dfb::exps;
    constexpr auto cb_recipsumexps = dfb::recip_sum_exps;
    constexpr auto cb_in0 = dfb::in0;
    constexpr auto cb_out0 = dfb::out0;
    DataflowBuffer cb_max_scaler_obj(cb_max_scaler);
    DataflowBuffer cb_sum_scaler_obj(cb_sum_scaler);
    DataflowBuffer cb_exps_obj(cb_exps);
    DataflowBuffer cb_recipsumexps_obj(cb_recipsumexps);
    DataflowBuffer cb_in0_obj(cb_in0);
    DataflowBuffer cb_out0_obj(cb_out0);
#if FUSED_SCALE_MASK
    // fused_scale/fused_attn/scale_mask are bound only on the fused scale-mask path.
    constexpr auto cb_fused_scale = dfb::fused_scale;
    constexpr auto cb_fused_attn = dfb::fused_attn;
    constexpr auto cb_scale_mask = dfb::scale_mask;
    DataflowBuffer cb_fused_scale_obj(cb_fused_scale);
    DataflowBuffer cb_fused_attn_obj(cb_fused_attn);
    DataflowBuffer cb_scale_mask_obj(cb_scale_mask);
#endif
#ifdef MASK_PADDED_DATA
    constexpr auto cb_mask_padded = dfb::mask_padded;
    DataflowBuffer cb_mask_padded_obj(cb_mask_padded);
#endif

    binary_op_init_common(cb_in0, cb_max_scaler, cb_exps);
#ifdef NUMERIC_STABLE
    constexpr auto cb_max = dfb::max;
#if defined(FUSED_SCALE_MASK) || defined(MASK_PADDED_DATA)
    // cb_x is a distinct intermediate (c_10) only on the numeric-stable paths that post-process a masked
    // buffer; otherwise the reads go straight from cb_in0 (see the calc_numeric_stable<cb_in0,...> call).
    constexpr auto cb_x = dfb::x;
    DataflowBuffer cb_x_obj(cb_x);
#endif
#else
    // Without numeric_stable, cb_x aliases cb_exps (Same-FIFO reuse) so exp results circulate in one buffer.
    constexpr auto cb_x = cb_exps;
    DataflowBuffer cb_x_obj(cb_x);
#endif

    cb_max_scaler_obj.wait_front(1);  // comes from the reader
    cb_sum_scaler_obj.wait_front(1);  // comes from the reader

#if FUSED_SCALE_MASK
    cb_fused_scale_obj.wait_front(1);
#endif

    constexpr int dst0 = 0;
    std::uint32_t ht = start_ht;
    bool wait_mask = true;
    for (std::uint32_t ncht = 0; ncht < NCHt; ncht++) {
#if FUSED_SCALE_MASK
        reconfig_data_format(cb_in0, cb_fused_scale);
        pack_reconfig_data_format(cb_scale_mask);
        mul_tiles_bcast_scalar_init_short(cb_in0, cb_fused_scale);
        for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
            // apply fused scale [*= 1/sqrt(...)]
            tile_regs_acquire();
            cb_in0_obj.wait_front(ndst);
            cb_scale_mask_obj.reserve_back(ndst);
            for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, wt8, 0, wt8);  // mul bcast-HW -> DST[wt8]
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                pack_tile(wt8, cb_scale_mask);  // reuse exps buffer
            }
            tile_regs_release();
            cb_scale_mask_obj.push_back(ndst);
            cb_in0_obj.pop_front(ndst);
        }
        reconfig_data_format(cb_scale_mask, cb_fused_attn);

#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif

#ifdef CAUSAL_MASK
        add_tiles_init(cb_scale_mask, cb_fused_attn);
#else
        add_bcast_rows_init_short(cb_scale_mask, cb_fused_attn);
#endif
        for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
            tile_regs_acquire();
            cb_scale_mask_obj.wait_front(ndst);
#ifdef CAUSAL_MASK
            cb_fused_attn_obj.wait_front(wt + ndst);  // cumulative wait for up to Wt tiles
            for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                add_tiles(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#else
            if (wait_mask) {
                cb_fused_attn_obj.wait_front(wt + ndst);  // cumulative wait for up to Wt tiles, only at first ht
            }

            for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#endif
            cb_scale_mask_obj.pop_front(ndst);
            cb_x_obj.reserve_back(ndst);
#ifndef NUMERIC_STABLE
            for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                pack_tile(wt8, cb_x);  // reuse the exps buffer again, this time in a circular manner
            }
            tile_regs_release();
            cb_x_obj.push_back(ndst);
        }

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        calc_numeric_stable<cb_x, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
#endif

#ifdef CAUSAL_MASK
        cb_fused_attn_obj.pop_front(Wt);
#else
        if (wait_mask) {
            wait_mask = false;
        }
        ht++;
        if (ht == Ht) {
            cb_fused_attn_obj.pop_front(Wt);
            ht = 0;
            wait_mask = true;
        }
#endif  // CAUSAL_MASK

        reconfig_data_format(cb_exps, cb_sum_scaler);
#else
        reconfig_data_format(cb_in0, cb_in0);
        pack_reconfig_data_format(cb_exps);
        copy_tile_to_dst_init_short(cb_in0);  // need to copy from CB to DST to be able to run sfpu math
#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
#ifdef MASK_PADDED_DATA
        {
            for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
                tile_regs_acquire();
                cb_in0_obj.wait_front(ndst);
                for (std::uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    if (wt == (Wt - ndst) && (wt8 == ndst - 1)) {
                        reconfig_data_format(cb_in0, cb_mask_padded);
                        add_bcast_rows_init_short(cb_in0, cb_mask_padded);
                        cb_mask_padded_obj.wait_front(1);
                        add_tiles_bcast_rows(cb_in0, cb_mask_padded, wt8, 0, wt8);
                    } else {
                        copy_tile(cb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
                    }
                }
                cb_in0_obj.pop_front(ndst);

                cb_x_obj.reserve_back(ndst);
#ifndef NUMERIC_STABLE
                for (std::uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
                }
#endif
                tile_regs_commit();
                tile_regs_wait();
                for (std::uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    pack_tile(wt8, cb_x);  // DST[0]->cb_id[wt]
                }
                tile_regs_release();
                cb_x_obj.push_back(ndst);
            }

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<cb_x, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
#endif
        }
#else
        {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<cb_in0, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
#else
            for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
                tile_regs_acquire();
                cb_in0_obj.wait_front(ndst);
                for (std::uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    copy_tile(cb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
                }
                cb_in0_obj.pop_front(ndst);

                cb_exps_obj.reserve_back(ndst);
                for (std::uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
                }
                tile_regs_commit();
                tile_regs_wait();
                for (std::uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    pack_tile(wt8, cb_exps);  // DST[0]->cb_id[wt]
                }
                tile_regs_release();
                cb_exps_obj.push_back(ndst);
            }
#endif
        }
#endif  // MASK_PADDED_DATA
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal post-processing (1/sum)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(Wt),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t) {
                recip_tile_init();
                recip_tile(0);
            });

        cb_recipsumexps_obj.wait_front(1);  // will reuse Wt times for bcast

        reconfig_data_format(cb_exps, cb_recipsumexps);
        pack_reconfig_data_format(cb_out0);
        // now cb_sumexps has exp tiles, need to multiply by our DST[2]
        // by now we already did a cumulative wait for Wt tiles in cb_exps
        mul_bcast_cols_init_short(cb_exps, cb_recipsumexps);
        for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
            tile_regs_acquire();
            cb_out0_obj.reserve_back(ndst);
            for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                // wt+wt8 since we pop Wt after the entire loop
                mul_tiles_bcast<BroadcastType::COL>(
                    cb_exps, cb_recipsumexps, wt + wt8, 0, wt8);  // tile *= 1/(sum(exp(x)))
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                pack_tile(wt8, cb_out0);
            }
            tile_regs_release();
            cb_out0_obj.push_back(ndst);
        }
        cb_recipsumexps_obj.pop_front(1);
        cb_exps_obj.pop_front(Wt);
    }  // NCHt loop
    // The scaler tiles are each waited once and reused across the whole NCHt loop; pop them at
    // the end so the CBs are left balanced.
    cb_max_scaler_obj.pop_front(1);
    cb_sum_scaler_obj.pop_front(1);
#if FUSED_SCALE_MASK
    cb_fused_scale_obj.pop_front(1);
#endif
}
