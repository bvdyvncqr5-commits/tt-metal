# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import run_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.skipif(bool(os.getenv("TT_METAL_SIMULATOR")), reason="Llama-3.1-8B shapes need hardware")

FIDELITY = {"HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}

PREFILL = [
    (1, 128, 4096, 6144, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", True, (8, 10), 1, 1, 1, 28, True),
    (1, 128, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, "HiFi2", True, (8, 8), 8, 1, 1, 19, True),
    (1, 128, 4096, 14336, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16, "LoFi", False, (8, 8), 8, 4, 1, 64, False),
    (1, 128, 14336, 4096, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", False, (8, 8), 8, 1, 1, 19, False),
    (1, 1024, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, "HiFi2", True, (8, 8), 8, 1, 4, 19, True),
    (2, 512, 4096, 14336, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16, "LoFi", False, (8, 8), 4, 4, 2, 64, False),
]
MINIMAL = [
    (1, 1024, 4096, 6144, ttnn.bfloat16, True, (8, 10)),
    (2, 512, 14336, 4096, ttnn.bfloat8_b, False, (8, 8)),
]
DECODE = [
    (32, 4096, 16032, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, "HiFi2", False, (8, 8), 2, 8),
    (32, 4096, 14336, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16, "LoFi", False, (8, 8), 2, 7),
    (32, 14336, 4096, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", False, (8, 8), 7, 2),
    (32, 4096, 6144, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", True, (8, 4), 4, 6),
    (32, 4096, 4096, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", True, (8, 4), 4, 4),
]


def dram_width_sharded(device, k, n):
    g = device.dram_grid_size()
    padded = -(-n // (32 * g.x)) * 32 * g.x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    spec = ttnn.ShardSpec(grid, (k, padded // g.x), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def kernel_config(fidelity, fp32):
    return ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE,
        math_fidelity=FIDELITY[fidelity],
        math_approx_mode=fp32,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


def pcc_for(fidelity, x_dtype):
    return 0.99 if fidelity == "LoFi" or x_dtype == ttnn.bfloat8_b else 0.999


def check(device, x, w_dtype, k, n, run, pcc):
    x_ref = ttnn.to_torch(x).float()
    for seed in (0, 1):
        torch.manual_seed(seed)
        w = ttnn.from_torch(
            torch.randn(1, 1, k, n) * 0.02,
            dtype=w_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_width_sharded(device, k, n),
        )
        out = ttnn.to_torch(run(w)).float()
        expected = x_ref @ ttnn.to_torch(w).float()[..., :k, :n]
        assert_with_pcc(expected, out[..., : expected.shape[-1]], pcc)
        ttnn.deallocate(w)


@run_for_blackhole()
@pytest.mark.parametrize(
    "batch, m, k, n, x_dtype, w_dtype, out_dtype, fidelity, fp32, grid, in0_block_w, sbw, per_core_m, per_core_n, fuse_batch",
    PREFILL,
)
def test_llama_prefill_matmul_shapes(
    device,
    batch,
    m,
    k,
    n,
    x_dtype,
    w_dtype,
    out_dtype,
    fidelity,
    fp32,
    grid,
    in0_block_w,
    sbw,
    per_core_m,
    per_core_n,
    fuse_batch,
):
    torch.manual_seed(1)
    x = ttnn.from_torch(
        torch.randn(1, batch, m, k),
        dtype=x_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sbw,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=fuse_batch,
    )
    ck = kernel_config(fidelity, fp32)

    def run(w):
        return ttnn.linear(
            x,
            w,
            program_config=config,
            compute_kernel_config=ck,
            dtype=out_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    check(device, x, w_dtype, k, n, run, pcc_for(fidelity, x_dtype))


@run_for_blackhole()
@pytest.mark.parametrize("batch, m, k, n, x_dtype, fp32, grid", MINIMAL)
def test_llama_prefill_minimal_matmul_shapes(device, batch, m, k, n, x_dtype, fp32, grid):
    torch.manual_seed(1)
    x = ttnn.from_torch(
        torch.randn(1, batch, m, k),
        dtype=x_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=1,
        subblock_w=1,
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
    )
    ck = kernel_config("HiFi2", fp32)

    def run(w):
        return ttnn.experimental.minimal_matmul(x, w, config=config, compute_kernel_config=ck)

    check(device, x, ttnn.bfloat8_b, k, n, run, pcc_for("HiFi2", x_dtype))


@run_for_blackhole()
@pytest.mark.parametrize(
    "m, k, n, x_dtype, w_dtype, out_dtype, fidelity, fp32, act_grid, in0_block_w, per_core_n", DECODE
)
def test_llama_decode_matmul_shapes(
    device, m, k, n, x_dtype, w_dtype, out_dtype, fidelity, fp32, act_grid, in0_block_w, per_core_n
):
    torch.manual_seed(1)
    cores = act_grid[0] * act_grid[1]
    act_mem = ttnn.create_sharded_memory_config(
        (m, k // cores),
        core_grid=ttnn.CoreGrid(y=act_grid[1], x=act_grid[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, m, k), dtype=x_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=act_mem
    )
    config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w, per_core_M=1, per_core_N=per_core_n, fused_activation=None
    )
    ck = kernel_config(fidelity, fp32)

    def run(w):
        return ttnn.linear(
            x,
            w,
            program_config=config,
            compute_kernel_config=ck,
            dtype=out_dtype,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

    check(device, x, w_dtype, k, n, run, pcc_for(fidelity, x_dtype))


@run_for_blackhole()
@pytest.mark.parametrize("k, n, x_dtype, fp32, grid", [(4096, 6144, ttnn.bfloat16, True, (8, 10))])
def test_llama_minimal_matmul_rows_order_independent(device, k, n, x_dtype, fp32, grid):
    torch.manual_seed(0)
    w = ttnn.from_torch(
        torch.randn(1, 1, k, n) * 0.02,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_width_sharded(device, k, n),
    )
    config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=1,
        subblock_w=1,
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
    )
    ck = kernel_config("HiFi2", fp32)
    inputs = {}
    for m in (128, 1024, 2048):
        torch.manual_seed(m)
        inputs[m] = torch.randn(1, 1, m, k)
    outs = {}
    new_programs = {}
    for order in ((1024, 128, 2048), (128, 2048, 1024)):
        device.clear_program_cache()
        for m in order:
            x = ttnn.from_torch(
                inputs[m], dtype=x_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            before = device.num_program_cache_entries()
            out = ttnn.experimental.minimal_matmul(x, w, config=config, compute_kernel_config=ck)
            new_programs[(order[0], m)] = device.num_program_cache_entries() - before
            out = ttnn.to_torch(out)
            ttnn.deallocate(x)
            if m in outs:
                assert torch.equal(outs[m], out)
            else:
                expected = inputs[m].to(torch.bfloat16).float() @ ttnn.to_torch(w).float()[..., :k, :n]
                assert_with_pcc(expected, out.float()[..., :n], 0.999)
                outs[m] = out
    if device.dram_grid_size().x == 7:
        assert new_programs[(128, 1024)] > 0
