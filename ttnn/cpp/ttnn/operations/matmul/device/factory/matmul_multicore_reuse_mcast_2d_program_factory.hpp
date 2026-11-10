// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct Mcast2DReadOptions {
    uint32_t buffer_depth = 0;
    bool coalesced_read = false;
    bool in0_resident = false;
    bool in1_dram_sharded_per_n_block = false;
    bool out_flush_per_block = false;
};

struct MatmulMultiCoreReuseMcast2DProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts_with_read_options(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const Mcast2DReadOptions& read_options);
};

}  // namespace ttnn::prim
