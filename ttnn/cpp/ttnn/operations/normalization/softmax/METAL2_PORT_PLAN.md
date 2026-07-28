# Port Plan — `normalization/softmax`

Port plan for `SoftmaxDeviceOperation`, ported from the `descriptor` (`ProgramDescriptor`)
factory concept to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Atomic-unit decomposition (7 units)

Each `ProgramFactory` (+ the kernel entry points it binds) is one atomic port unit.

| # | Unit | factory | kernels | coupling |
|---|---|---|---|---|
| 1-5 | **General W-small/W-large/H-small/H-large/C-large** | `SoftmaxProgramFactoryGeneral{WSmall,WLarge,HSmall,HLarge,CLarge}` | borrowed from `moreh/moreh_softmax/device/kernels/` (one trio each) | **co-port with `moreh/moreh_softmax`** (issue #51081) — shared kernel rewrite must land in both ops at once |
| 6 | **Attention interleaved** | `SoftmaxProgramFactoryAttentionOptimized` | own `device/kernels/attention/` (2 readers + 1 writer + 2 compute, runtime source selection small/large) | independent |
| 7 | **Attention sharded** | `SoftmaxShardedProgramFactoryAttentionOptimized` | own `device/kernels/attention/` (3 readers + 1 compute, no writer) | independent |

The five General units share their kernel trios with `moreh/moreh_softmax` — see
`ttnn/cpp/ttnn/operations/moreh/moreh_softmax/METAL2_PORT_PLAN.md` for the shared kernel
inventory, DFB naming, hw_config, and `unpack_modes` handling. The General and moreh
factories are near-clones (same CBs, same kernels, same RTA/CTA layout); they differ only
in resolved data formats (norm: intermed=`fp32?F32:F16_b`, mask/scaler=`bfp8?F16_b:df`) and
defines. Both factories bind the shared kernel with **identical accessor/arg names**.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (all 7 factories are `static ProgramDescriptor create_descriptor(...)`, device op hpp:25-57).
- Variants: 7 (5 General + 2 Attention). `select_program_factory` (device_operation.cpp:120) picks by op type / dim / rank / sharding / L1-fit.
- Custom `compute_program_hash`: none.
- Op-owned tensors: none.

### General units — see the moreh plan (identical shape). Norm-specific deltas:
- `intermed_data_format = fp32 ? Float32 : Float16_b` (moreh uses `data_format` when not fp32).
- `mask_scaler_format = (data_format==Bfp8_b) ? Float16_b : data_format`.
- compute defines: `SOFTMAX=1`; `FP32_DEST_ACC_EN=1` when `fp32_dest_acc_en || bfp8`.
- Kernel path prefix `SOFTMAX_KERNEL_PATH_GENERAL` = `moreh/moreh_softmax/device/kernels`.

### Attention interleaved (unit 6) — inventory
Runtime source selection (small vs large by L1 fit): reader ∈ {`reader_unary_interleaved_sm.cpp`, `reader_unary_interleaved_sm_large_tensor.cpp`}; compute ∈ {`compute/softmax.cpp`, `compute/softmax_large_tensor.cpp`}; writer = `writer_unary_interleaved_start_id_blocked_sm.cpp` (shared across both paths). All 5 sources convert together (atomic).

Compile-time axes: `FUSED_SCALE_MASK` (mask present), `CAUSAL_MASK`, `NUMERIC_STABLE`; runtime/host: `mask_padded_data` (host-known: W>W_unpadded), `use_large_kernel` (host L1 pressure).

CBs (conditional allocation): c_0 in0, c_11 out0, c_7 recipsumexps, c_2 max_scaler, c_13 sum_scaler, c_6 exps (always); c_9 scale_mask, c_3 fused_scale, c_4 fused_attn (FUSED); c_5 pad_mask (allocated always, touched only when mask_padded_data); c_8 max (NUMERIC_STABLE); c_10 cb_x (NUMERIC_STABLE&&(mask||pad) OR large); c_12/c_15/c_16 (large kernel).

Endpoint census (from brief, re-derived): 1P+1C R→C: c_0, c_2, c_13, c_3(FUSED), c_4(FUSED); W→C: c_5 (only when mask_padded_data — conditional binding). C→W: c_11. Self-loop (compute-internal): c_6, c_7, c_9(FUSED), c_8(NS), c_10, c_12/c_15/c_16(large).

Reader RTA (non-causal): `{src_addr, blk, pre_scale, num_blks, tile_offset, Wt, Ht, mask_addr, start_ht, start_mask_id, in0_t}` — src_addr→`tensor::src`, mask_addr→`tensor::mask` (FUSED, conditional); rest named; `in0_t` unused by small reader (dead RTA), `Ht/start_ht/start_mask_id/pre_scale` FUSED-only. Causal adds `mask_start_ht, mask_offset`. Reader CTA: `TensorAccessorArgs(src)` [+ `TensorAccessorArgs(mask)` FUSED] [+ `num_tiles_causal_mask` CAUSAL].
Writer RTA: `{dst_addr, num_tiles, tile_offset, blk, mask_padded_data}` — dst→`tensor::dst`; CTA `{num_datum_padded, tile_hw}` + `TensorAccessorArgs(dst)`.
Compute RTA: `{NCHt, Ht, Wt, ndst, start_ht, mask_padded_data, cb_length}` — `cb_length` unused by small compute (dead), used by large.

**Watch-for (unit 6):** `mask_padded_data` gates the c_5 producer(writer)/consumer(compute) binding. It is host-known → promote to a `MASK_PADDED_DATA` `#define` on writer+compute and conditionally bind c_5 (Conditional/optional DFB pattern). `cb_x` no-mask path aliases `cb_exps` (Same-FIFO aliasing); NUMERIC_STABLE path uses c_10 — `#ifdef`-gate the c_10 handle/uses.

### Attention sharded (unit 7) — inventory to complete when unit is reached
Borrowed-memory DFBs (c_0 input, c_11 output, c_3 sharded-mask via `.buffer=`), interleaved-mask Case-1. Readers: `reader_unary_sharded_sm.cpp` / `..._causal_mask_hw_dims.cpp` / `..._rm_mask.cpp` (runtime selection); compute `softmax_sharded.cpp`; no writer.

### Semaphores
none (whole op).

### Cross-op kernels
General units borrow all kernels from `moreh/moreh_softmax/device/kernels/` — co-migrated in this PR (report → Open items). Attention kernels are owned; no borrow.

### Flags
- `select_program_factory` has unreachable `return` after `std::visit` (device_operation.cpp:140,154) — dead code, off-limits, route to report.
- Attention interleaved creates kernels on unused cores (`all_device_cores`, zero-args for `i>=num_cores`) — existing behavior; the Metal 2.0 port derives placement from `WorkUnitSpec::target_nodes = all_cores` (the used set), which is the faithful & cleaner shape. Note in report.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (plain).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: mixed-concept `program_factory_t` variant during incremental port (ported factories → `create_program_artifacts`; rest keep `create_descriptor`). Per-factory hpp declaration flips.

## Planned Spec Shape
- **General units**: identical to moreh (see moreh plan) — 10 DFBs, input/output TensorParameters, reader/writer/compute_g1[/g2], wu_g1/wu_g2.
- **Attention interleaved**: DFBs per the conditional census above; TensorParameters input/output[/mask FUSED]; reader/writer/compute KernelSpecs (source chosen by `use_large_kernel`); single WorkUnitSpec over `all_cores` (the used set from work-split; reader+writer+compute co-resident — g1/g2 differ only by compute RTA not CTA here, so **one** compute KernelSpec suffices — compute takes per-core counts as RTA in legacy, not CTA, so no multiplicity to preserve).
- **Attention sharded**: borrowed-memory DFBs (`borrowed_from`), TensorParameters input/output[/mask]; reader/compute; WorkUnitSpec over the shard grid.

## Preserved Multiplicity
| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs |
|---|---|---|---|
| General: compute g1+g2 over disjoint groups | COMPUTE_G1, COMPUTE_G2 | wu_g1, wu_g2 | in/mask/scalers CONSUMER, out PRODUCER (single-role, disjoint nodes) |
| Attention interleaved/sharded: single compute descriptor (per-core work via RTA) | one COMPUTE | one WU | 1:1 |

## Dropped Plumbing
Per the General moreh table, plus for attention: `TensorAccessorArgs(src/mask/dst)`→TensorBindings; `src_addr`/`mask_addr`/`dst_addr` buffer RTAs→TensorBindings; positional CTAs→named; `get_tile_size(cb)`→`dfb.get_entry_size()` (DM) / `dfb.get_tile_size()` (compute).

## Applied Patterns
- General: see moreh plan (Same-FIFO alias, self-loop intermediates, avoided CTA→RTA demotion, Style-A hw_config, fp32 `unpack_modes`).
- Attention interleaved: [Conditional/optional DFB bindings] for c_5 pad-mask (host `mask_padded_data`→define) and FUSED CBs (c_3/c_4/c_9) + mask tensor; [Same-FIFO aliasing] for `cb_x`=`cb_exps` no-mask path; [Self-loop] for compute intermediates.
- Attention sharded: borrowed-memory DFBs (`borrowed_from`), self-loop for compute-only borrowed CBs.

## Deferred / Flagged
- General units gated on co-porting `moreh/moreh_softmax`'s matching factory in the same PR.
- Attention interleaved is the highest-complexity factory (5 interacting axes, conditional CBs); ported last.
- **General W-large ported, with a `noinline` workaround.** Its shared compute kernel (`moreh_softmax_w_large.cpp`) initially failed to JIT-compile in the fp32 path (out-of-scope LLK addrmod cliff; confirmed a port regression), now worked around by a `noinline` split in that kernel. See `METAL2_PORT_REPORT.md` and the moreh op's report. So the ported General set is **W-small, W-large, H-small, H-large, C-large** (5 of 5); the 2 attention factories remain on legacy. The durable fix belongs upstream in the LLK.
