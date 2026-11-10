// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <variant>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct BhMatmulRoute {
    operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig program_config;
    Mcast1DReadOptions read_options;
};

struct BhMatmul2DRoute {
    operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config;
    Mcast2DReadOptions read_options;
};

struct BhMatmulDramShardedRoute {
    operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig program_config;
    DramShardedReadOptions read_options;
};

namespace bh_matmul_route_detail {

using tt::tt_metal::DataType;
using tt::tt_metal::MathFidelity;

struct Row {
    uint32_t M, K, N;
    DataType activation_dtype;
    DataType weight_dtype;
    uint32_t in0_block_w;
    MathFidelity math_fidelity;
    bool fp32_dest_acc_en;
    DataType output_dtype;
    uint32_t model_grid_x, model_grid_y;
    uint32_t grid_x, grid_y;
    uint32_t per_core_n_mult;
    uint32_t out_subblock_h;
    bool max_subblock_w;
    Mcast1DReadOptions read_options;
};

// clang-format off
inline const std::array<Row, 4>& rows() {
    static const std::array<Row, 4> table = {{
        {128, 4096, 6144, DataType::BFLOAT16, DataType::BFLOAT8_B, 1, MathFidelity::HiFi2, true, DataType::BFLOAT16, 8, 10, 11, 10, 1, 1, false,
         {.in1_read_by_page = true, .in1_coalesced_read = true, .in0_group = 8}},
        {128, 4096, 4096, DataType::BFLOAT8_B, DataType::BFLOAT8_B, 8, MathFidelity::HiFi2, true, DataType::BFLOAT8_B, 8, 8, 11, 10, 1, 1, false,
         {.in1_read_by_page = true, .in1_coalesced_read = true}},
        {128, 4096, 14336, DataType::BFLOAT16, DataType::BFLOAT4_B, 8, MathFidelity::LoFi, false, DataType::BFLOAT16, 8, 8, 11, 8, 2, 1, true,
         {.in1_read_by_page = true, .in1_coalesced_read = true, .in1_buffer_depth = 3}},
        {128, 14336, 4096, DataType::BFLOAT8_B, DataType::BFLOAT8_B, 8, MathFidelity::HiFi2, false, DataType::BFLOAT16, 8, 8, 11, 9, 1, 4, false,
         {.in1_read_by_page = true, .in1_coalesced_read = true, .out_flush_per_block = true, .in1_buffer_depth = 4}},
    }};
    return table;
}

struct Row2D {
    uint32_t batch, M, K, N;
    DataType activation_dtype;
    DataType weight_dtype;
    uint32_t in0_block_w;
    MathFidelity math_fidelity;
    bool fp32_dest_acc_en;
    DataType output_dtype;
    uint32_t model_grid_x, model_grid_y;
    uint32_t grid_x, grid_y;
    uint32_t per_core_M, per_core_N;
    uint32_t out_block_h, out_block_w;
    uint32_t out_subblock_h, out_subblock_w;
    Mcast2DReadOptions read_options;
};

inline const std::array<Row2D, 2>& rows_2d() {
    static const std::array<Row2D, 2> table = {{
        {1, 1024, 4096, 4096, DataType::BFLOAT8_B, DataType::BFLOAT8_B, 8, MathFidelity::HiFi2, true, DataType::BFLOAT8_B,
         8, 8, 11, 10, 3, 13, 3, 13, 3, 1, {.coalesced_read = true}},
        {2, 512, 4096, 14336, DataType::BFLOAT16, DataType::BFLOAT4_B, 4, MathFidelity::LoFi, false, DataType::BFLOAT16,
         8, 8, 11, 10, 3, 45, 3, 15, 1, 5,
         {.buffer_depth = 6,
          .coalesced_read = true,
          .in0_resident = true,
          .in1_dram_sharded_per_n_block = true,
          .out_flush_per_block = true}},
    }};
    return table;
}

struct DramShardedRow {
    uint32_t M, K, N;
    DataType activation_dtype;
    DataType weight_dtype;
    uint32_t in0_block_w, per_core_N;
    MathFidelity math_fidelity;
    bool fp32_dest_acc_en;
    DataType output_dtype;
    uint32_t workers_per_bank;
    DramShardedReadOptions read_options;
};

inline const std::array<DramShardedRow, 5>& dram_sharded_rows() {
    using Vc = DramShardedReadOptions::ReaderVc;
    static const std::array<DramShardedRow, 5> table = {{
        {32, 4096, 16032, DataType::BFLOAT16, DataType::BFLOAT8_B, 2, 8, MathFidelity::HiFi2, false, DataType::BFLOAT8_B, 2,
         {.uneven_readers = true, .reader_vc = Vc::LeastUsedInRow, .streamed_write_back = true, .round_robin_readers = true}},
        {32, 4096, 14336, DataType::BFLOAT16, DataType::BFLOAT4_B, 2, 7, MathFidelity::LoFi, false, DataType::BFLOAT16, 2,
         {.uneven_readers = true, .reader_vc = Vc::LeastUsedInRow, .streamed_write_back = true}},
        {32, 14336, 4096, DataType::BFLOAT8_B, DataType::BFLOAT8_B, 7, 2, MathFidelity::HiFi2, false, DataType::BFLOAT16, 2,
         {.uneven_readers = true, .reader_vc = Vc::BankPlusReader, .streamed_write_back = true}},
        {32, 4096, 6144, DataType::BFLOAT16, DataType::BFLOAT8_B, 4, 6, MathFidelity::HiFi2, true, DataType::BFLOAT16, 3,
         {.uneven_readers = true, .reader_vc = Vc::BankPlusReader, .streamed_write_back = true}},
        {32, 4096, 4096, DataType::BFLOAT16, DataType::BFLOAT8_B, 4, 4, MathFidelity::HiFi2, true, DataType::BFLOAT16, 3,
         {.uneven_readers = true, .reader_vc = Vc::BankPlusReader, .streamed_write_back = true}},
    }};
    return table;
}

// clang-format on

inline uint32_t largest_divisor_at_most(uint32_t n, uint32_t limit) {
    for (uint32_t d = std::min(n, limit); d > 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

inline bool p100_device(const ttnn::Tensor& a) {
    const auto grid = a.device()->compute_with_storage_grid_size();
    return grid.x == 11 && grid.y == 10 && a.device()->num_dram_channels() == 7;
}

inline bool plain_two_input_matmul(const MatmulParams& params, const MatmulInputs& tensors) {
    if (params.untilize_out || params.transpose_a || params.transpose_b || params.user_run_batched ||
        params.global_cb.has_value() || params.sub_device_id.has_value() || !params.prefetcher_pipes.empty() ||
        params.in0_column_offset.has_value() || params.user_fused_activation.has_value() ||
        !params.compute_kernel_config.has_value()) {
        return false;
    }
    if (params.output_tile.has_value() && (params.output_tile->get_tile_shape()[0] != tt::constants::TILE_HEIGHT ||
                                           params.output_tile->get_tile_shape()[1] != tt::constants::TILE_WIDTH)) {
        return false;
    }
    if (tensors.input_tensors.size() != 2) {
        return false;
    }
    for (const auto& t : tensors.optional_input_tensors) {
        if (t.has_value()) {
            return false;
        }
    }
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    return a.device()->arch() == tt::ARCH::BLACKHOLE && p100_device(a) && a.layout() == Layout::TILE &&
           b.layout() == Layout::TILE && b.memory_config().is_dram() &&
           b.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
}

inline bool full_rect_or_unset(const std::optional<tt::tt_metal::CoreRangeSet>& cores, const CoreCoord& grid) {
    return !cores.has_value() ||
           cores.value() == tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(
                                tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1)));
}

struct InterleavedShape {
    uint32_t batch, M, K, N;
};

inline std::optional<InterleavedShape> interleaved_shape(const MatmulParams& params, const MatmulInputs& tensors) {
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    if (a.memory_config().is_sharded() || !a.memory_config().is_dram() || params.output_mem_config.is_sharded()) {
        return std::nullopt;
    }
    const auto& as = a.padded_shape();
    const auto& bs = b.padded_shape();
    if (as.rank() < 2 || bs.rank() < 2 || a.logical_shape() != as || b.logical_shape() != bs) {
        return std::nullopt;
    }
    uint32_t batch = 1;
    for (int i = 0; i < static_cast<int>(as.rank()) - 2; ++i) {
        batch *= as[i];
    }
    for (int i = 0; i < static_cast<int>(bs.rank()) - 2; ++i) {
        if (bs[i] != 1) {
            return std::nullopt;
        }
    }
    if (bs[-2] != as[-1]) {
        return std::nullopt;
    }
    return InterleavedShape{
        batch, static_cast<uint32_t>(as[-2]), static_cast<uint32_t>(as[-1]), static_cast<uint32_t>(bs[-1])};
}

}  // namespace bh_matmul_route_detail

inline std::optional<BhMatmulRoute> bh_matmul_route(const MatmulParams& params, const MatmulInputs& tensors) {
    using namespace bh_matmul_route_detail;
    if (!params.program_config.has_value() || !plain_two_input_matmul(params, tensors)) {
        return std::nullopt;
    }
    const auto* cfg =
        std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(&params.program_config.value());
    if (cfg == nullptr || cfg->transpose_mcast || cfg->fused_activation.has_value() ||
        !full_rect_or_unset(cfg->allowed_worker_cores, cfg->compute_with_storage_grid_size)) {
        return std::nullopt;
    }
    const auto shape = interleaved_shape(params, tensors);
    if (!shape.has_value() || shape->batch != 1) {
        return std::nullopt;
    }
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), params.compute_kernel_config.value());
    (void)math_approx_mode;
    if (!packer_l1_acc || dst_full_sync_en) {
        return std::nullopt;
    }
    const uint32_t M = shape->M, K = shape->K, N = shape->N;
    for (const auto& r : rows()) {
        if (r.M != M || r.K != K || r.N != N || r.activation_dtype != a.dtype() || r.weight_dtype != b.dtype() ||
            r.in0_block_w != cfg->in0_block_w || r.math_fidelity != math_fidelity ||
            r.fp32_dest_acc_en != fp32_dest_acc_en || r.output_dtype != params.output_dtype.value_or(a.dtype())) {
            continue;
        }
        if (cfg->compute_with_storage_grid_size.x != r.model_grid_x ||
            cfg->compute_with_storage_grid_size.y != r.model_grid_y) {
            return std::nullopt;
        }
        const uint32_t Mt = M / tt::constants::TILE_HEIGHT;
        const uint32_t Nt = N / tt::constants::TILE_WIDTH;
        const uint32_t cores = r.grid_x * r.grid_y;
        const uint32_t per_core_N = ((Nt + cores - 1) / cores) * r.per_core_n_mult;
        if (Mt % r.out_subblock_h != 0) {
            return std::nullopt;
        }
        const uint32_t dest_tiles = (fp32_dest_acc_en ? 4u : 8u) / r.out_subblock_h;
        const uint32_t out_subblock_w = r.max_subblock_w ? largest_divisor_at_most(per_core_N, dest_tiles) : 1;
        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig pc{
            .compute_with_storage_grid_size = tt::tt_metal::CoreCoord(r.grid_x, r.grid_y),
            .in0_block_w = r.in0_block_w,
            .out_subblock_h = r.out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .out_block_h = Mt,
            .out_block_w = per_core_N,
            .per_core_M = Mt,
            .per_core_N = per_core_N,
            .fuse_batch = true,
            .fused_activation = std::nullopt,
            .mcast_in0 = true,
            .gather_in0 = false,
            .hop_cores = {},
            .num_global_cb_receivers = 0,
            .untilize_out = false,
            .allowed_worker_cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(
                tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(r.grid_x - 1, r.grid_y - 1))),
        };
        return BhMatmulRoute{.program_config = pc, .read_options = r.read_options};
    }
    return std::nullopt;
}

inline std::optional<BhMatmul2DRoute> bh_matmul_route_2d(const MatmulParams& params, const MatmulInputs& tensors) {
    using namespace bh_matmul_route_detail;
    if (!params.program_config.has_value() || !plain_two_input_matmul(params, tensors)) {
        return std::nullopt;
    }
    const auto* cfg =
        std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(&params.program_config.value());
    if (cfg == nullptr || cfg->transpose_mcast || cfg->fused_activation.has_value() ||
        !full_rect_or_unset(cfg->allowed_worker_cores, cfg->compute_with_storage_grid_size)) {
        return std::nullopt;
    }
    const auto shape = interleaved_shape(params, tensors);
    if (!shape.has_value()) {
        return std::nullopt;
    }
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), params.compute_kernel_config.value());
    (void)math_approx_mode;
    if (!packer_l1_acc || dst_full_sync_en) {
        return std::nullopt;
    }
    for (const auto& r : rows_2d()) {
        if (r.batch != shape->batch || r.M != shape->M || r.K != shape->K || r.N != shape->N ||
            r.activation_dtype != a.dtype() || r.weight_dtype != b.dtype() || r.in0_block_w != cfg->in0_block_w ||
            r.math_fidelity != math_fidelity || r.fp32_dest_acc_en != fp32_dest_acc_en ||
            r.output_dtype != params.output_dtype.value_or(a.dtype())) {
            continue;
        }
        if (cfg->compute_with_storage_grid_size.x != r.model_grid_x ||
            cfg->compute_with_storage_grid_size.y != r.model_grid_y) {
            return std::nullopt;
        }
        operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig pc{
            .compute_with_storage_grid_size = tt::tt_metal::CoreCoord(r.grid_x, r.grid_y),
            .in0_block_w = r.in0_block_w,
            .out_subblock_h = r.out_subblock_h,
            .out_subblock_w = r.out_subblock_w,
            .out_block_h = r.out_block_h,
            .out_block_w = r.out_block_w,
            .per_core_M = r.per_core_M,
            .per_core_N = r.per_core_N,
            .transpose_mcast = true,
            .fused_activation = std::nullopt,
            .fuse_batch = true,
        };
        return BhMatmul2DRoute{.program_config = pc, .read_options = r.read_options};
    }
    return std::nullopt;
}

inline std::optional<BhMatmulDramShardedRoute> bh_matmul_route_dram_sharded(
    const MatmulParams& params, const MatmulInputs& tensors) {
    using namespace bh_matmul_route_detail;
    if (!params.program_config.has_value() || !plain_two_input_matmul(params, tensors)) {
        return std::nullopt;
    }
    const auto* cfg = std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
        &params.program_config.value());
    if (cfg == nullptr || cfg->num_workers_per_dram_bank != 1 || cfg->use_multicore_pipeline ||
        cfg->fused_activation.has_value() || cfg->per_core_M != 1 || !params.output_mem_config.is_sharded()) {
        return std::nullopt;
    }
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    if (!a.memory_config().is_sharded() || a.memory_config().is_dram()) {
        return std::nullopt;
    }
    const auto& as = a.padded_shape();
    const auto& bs = b.padded_shape();
    if (as.rank() < 2 || bs.rank() < 2) {
        return std::nullopt;
    }
    for (int i = 0; i < static_cast<int>(as.rank()) - 2; ++i) {
        if (as[i] != 1) {
            return std::nullopt;
        }
    }
    for (int i = 0; i < static_cast<int>(bs.rank()) - 2; ++i) {
        if (bs[i] != 1) {
            return std::nullopt;
        }
    }
    const uint32_t M = as[-2], K = as[-1], N = b.logical_shape()[-1];
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), params.compute_kernel_config.value());
    (void)math_approx_mode;
    (void)packer_l1_acc;
    if (dst_full_sync_en) {
        return std::nullopt;
    }
    for (const auto& r : dram_sharded_rows()) {
        if (r.M != M || r.K != K || r.N != N || r.activation_dtype != a.dtype() || r.weight_dtype != b.dtype() ||
            r.in0_block_w != cfg->in0_block_w || r.per_core_N != cfg->per_core_N || r.math_fidelity != math_fidelity ||
            r.fp32_dest_acc_en != fp32_dest_acc_en || r.output_dtype != params.output_dtype.value_or(a.dtype())) {
            continue;
        }
        auto pc = *cfg;
        pc.num_workers_per_dram_bank = r.workers_per_bank;
        return BhMatmulDramShardedRoute{.program_config = pc, .read_options = r.read_options};
    }
    return std::nullopt;
}

struct MatmulBhMcast1DProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value) {
        auto route = bh_matmul_route(operation_attributes, tensor_args);
        TT_FATAL(route.has_value(), "MatmulBhMcast1DProgramFactory selected for a matmul it does not cover");
        MatmulParams routed = operation_attributes;
        routed.program_config = route->program_config;
        return MatmulMultiCoreReuseMcast1DProgramFactory::create_program_artifacts_with_read_options(
            routed, tensor_args, tensor_return_value, route->read_options);
    }
};

struct MatmulBhMcast2DProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value) {
        auto route = bh_matmul_route_2d(operation_attributes, tensor_args);
        TT_FATAL(route.has_value(), "MatmulBhMcast2DProgramFactory selected for a matmul it does not cover");
        MatmulParams routed = operation_attributes;
        routed.program_config = route->program_config;
        return MatmulMultiCoreReuseMcast2DProgramFactory::create_program_artifacts_with_read_options(
            routed, tensor_args, tensor_return_value, route->read_options);
    }
};

struct MatmulBhDramShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value) {
        auto route = bh_matmul_route_dram_sharded(operation_attributes, tensor_args);
        TT_FATAL(route.has_value(), "MatmulBhDramShardedProgramFactory selected for a matmul it does not cover");
        MatmulParams routed = operation_attributes;
        routed.program_config = route->program_config;
        return MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_program_artifacts_with_read_options(
            routed, tensor_args, tensor_return_value, route->read_options);
    }
};

}  // namespace ttnn::prim
