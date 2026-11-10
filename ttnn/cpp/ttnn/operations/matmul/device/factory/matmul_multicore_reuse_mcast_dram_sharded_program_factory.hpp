// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct DramShardedReadOptions {
    enum class ReaderVc : uint8_t { Default, BankPlusReader, LeastUsedInRow };
    bool uneven_readers = false;
    ReaderVc reader_vc = ReaderVc::Default;
    bool streamed_write_back = false;
    bool round_robin_readers = false;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts_with_read_options(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const DramShardedReadOptions& read_options);
};

}  // namespace ttnn::prim
