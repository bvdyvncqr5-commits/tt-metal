# Port Plan — `moreh/moreh_softmax`

Port plan for `MorehSoftmaxOperation`, ported from the `descriptor` (`ProgramDescriptor`)
factory concept to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

> **Port-together coupling (issue #51081).** This op *owns* the 15 kernels in
> `device/kernels/` that `normalization/softmax`'s five **General** factories borrow by
> file path. The kernel-side CB→DFB / named-token rewrite is a **single change** that must
> land across both ops at once. This plan is the *owner* half; see
> `ttnn/cpp/ttnn/operations/normalization/softmax/METAL2_PORT_PLAN.md` for the co-borrower half.
> The two ops are decomposed into **five shared kernel-trio units** (see below); each unit
> ports both ops' matching factory together with its one shared kernel trio.

## The 5 co-port units (this op's factories ↔ shared kernel trio ↔ norm General factory)

| Unit | moreh factory | norm General factory | shared kernel trio (`device/kernels/`) |
|---|---|---|---|
| **W-small** | `MorehSoftmaxWSmallFactory` | `SoftmaxProgramFactoryGeneralWSmall` | `reader_moreh_softmax_w` / `moreh_softmax_w` / `writer_moreh_softmax_w` |
| **W-large** | `MorehSoftmaxWLargeFactory` | `SoftmaxProgramFactoryGeneralWLarge` | `*_w_large` |
| **H-small** | `MorehSoftmaxHSmallFactory` | `SoftmaxProgramFactoryGeneralHSmall` | `reader_moreh_softmax_h` / `moreh_softmax_h` / `writer_moreh_softmax_h` |
| **H-large** | `MorehSoftmaxHLargeFactory` | `SoftmaxProgramFactoryGeneralHLarge` | `*_h_large` |
| **C-large** | `MorehSoftmaxCLargeFactory` | `SoftmaxProgramFactoryGeneralCLarge` | `*_c_large` |

Each moreh factory selects exactly **one fixed kernel trio** (no runtime source selection),
so each unit is a clean atomic port: 2 factories (one per op) + 3 shared kernels.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (each factory is `static ProgramDescriptor create_descriptor(...)`, device op hpp:49-62).
- Variants: 5 factories (WSmall/WLarge/HSmall/HLarge/CLarge), all interleaved. `select_program_factory` picks by strategy.
- Custom `compute_program_hash`: none — already default reflection-based hash (audit confirmed).
- Op-owned tensors: none (no `WorkloadDescriptor`, no `buffers` vector).

### Kernels (per unit; W-small shown, others structurally analogous)

**W-small** (`softmax_w_small/softmax_w_small.cpp`):
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `kernels/reader_moreh_softmax_w.cpp` | all_cores | `{is_fp32}` + `TensorAccessorArgs(input)` | `{input.buffer(), N, tile_offset, Wt, mask_w}` | — | Reader |
| writer | `kernels/writer_moreh_softmax_w.cpp` | all_cores | `TensorAccessorArgs(output)` | `{output.buffer(), N, tile_offset, Wt}` | — | Writer |
| compute_g1 | `kernels/moreh_softmax_w.cpp` | core_group_1 | `{N_g1, Wt}` | none | SOFTMAX/SOFTMIN/LOG, FP32_DEST_ACC_EN | Compute |
| compute_g2 | `kernels/moreh_softmax_w.cpp` | core_group_2 | `{N_g2, Wt}` | none | (same) | Compute |

### CBs (W-small — all unconditional; formats from each factory's own resolution)
| index | role | size (tiles) | data_format (moreh) |
|---|---|---|---|
| c_0 | input | Wt | data_format |
| c_1 | mask | 1 | data_format |
| c_2 | max scaler | 1 | data_format |
| c_3 | sum scaler | 1 | data_format |
| c_16 | output | Wt | data_format |
| c_24 | exp(x) | Wt | intermed (fp32?F32:df) |
| c_25 | reduce (1/sum) | 1 | intermed |
| c_26 | max | 1 | intermed |
| c_27 | x - max | Wt | intermed |
| c_28 | tmp | 1 | intermed |

### Semaphores
none.

### Tensor accessors
| host site | originating Tensor | RTA slot |
|---|---|---|
| reader `TensorAccessor(in_args, src_addr)` | `input` | RTA 0 (`input.buffer()`) |
| writer `TensorAccessor(out_args, dst_addr)` | `output` | RTA 0 (`output.buffer()`) |

### Work split
`split_work_to_cores_wt_core_range(core_range, num_kernel_rows)` → `(num_cores, all_cores, core_group_1, core_group_2, N_g1, N_g2)`. Two compute descriptors (g1/g2) over **disjoint** core groups.

### Cross-op kernels
The 15 `device/kernels/*.cpp` are **owned here** but co-instantiated by `normalization/softmax` General factories. In-place co-migration (both ops in this PR) per [Caution: Modifying a shared dataflow kernel]. Recorded in `METAL2_PORT_REPORT.md` → Open items.

### Flags
- `tensor_args_t` holds references (device op hpp:41-44) — CLAUDE.md rule 14 violation, but **off-limits** (op-level host code, not the factory body). Route to report; do not touch.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (plain — no op-owned tensors).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: `program_factory_t` variant mixes concepts during incremental port — a ported factory exposes `create_program_artifacts` (→ `ProgramSpecFactoryConcept`); un-ported factories keep `create_descriptor` (→ `ProgramDescriptorFactoryConcept`); `AllFactoriesValid` accepts the mix (each satisfies exactly one concept). Per-factory hpp declaration flips from `create_descriptor` to `create_program_artifacts`.

## Planned Spec Shape (per unit)
- **KernelSpecs**: reader, writer, compute_g1, and compute_g2 (when core_group_2 non-empty). Preserve the g1/g2 multiplicity (two KernelSpecs of the same compute source with per-group `N` CTA).
- **DataflowBufferSpecs**: one per CB (10 for W-small). No conditional/borrowed/aliased DFBs.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input` (bound by reader), `output` (bound by writer).
- **WorkUnitSpecs**: `wu_g1 = {reader, writer, compute_g1}` on core_group_1; `wu_g2 = {reader, writer, compute_g2}` on core_group_2 (when present). Reader/writer run on `all_cores` = union of the two groups via membership in both WUs.

## Preserved Multiplicity
| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| compute g1 + g2 (moreh_softmax_w.cpp) over disjoint core_group_1/2 | COMPUTE_G1, COMPUTE_G2 | wu_g1, wu_g2 | in/mask/max_scaler/sum_scaler CONSUMER; out PRODUCER — one role each (disjoint node sets → legal single-role, **no** multi-binding flag) |

## Dropped Plumbing (per unit)
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA slot ≥1 | `TensorAccessorArgs(input)` | `TensorBinding(input)` (`tensor::src`) |
| reader RTA slot 0 | `input.buffer()` | `TensorBinding(input)` |
| writer CTA | `TensorAccessorArgs(output)` | `TensorBinding(output)` (`tensor::dst`) |
| writer RTA slot 0 | `output.buffer()` | `TensorBinding(output)` |
| reader CTA slot 0 | positional `is_fp32` | named `args::is_fp32` |
| reader RTA 1-4 | positional | named `args::{num_rows,tile_offset,Wt,mask_w}` |
| writer RTA 1-3 | positional | named `args::{num_rows,tile_offset,Wt}` |
| compute CTA 0-1 | positional `{N, Wt}` | named `args::{N,Wt}` (per KernelSpec) |
| kernel magic CB ids (`tt::CBIndex::c_*`) | constexpr cb ids | `dfb::<name>` handles |
| `get_tile_size(cb_id)` (DM reader/writer) | free fn | `dfb.get_entry_size()` (DM-safe size query; == tile bytes) |

## Applied Patterns
- [Pass DFB handles directly to LLKs and kernel-lib helpers]: `dfb::name` into `reduce<..., cb_in0, ...>` NTTP, `sub_tiles_bcast(cb_in0,...)`, `binary_op_init_common(...)`, `mask_tile_to_cb(dfb_obj,...)`.
- [Same-FIFO aliasing]: `constexpr auto cb_in0 = dfb::in0;` gives the LLK-side `uint32_t` handle while `DataflowBuffer dfb_in0_obj(cb_in0)` gives the FIFO object — one DFB, both spellings.
- [Self-loop DFB binding]: compute-internal intermediates c_24/c_25/c_26/c_27/c_28 — compute bound PRODUCER+CONSUMER.
- [Demoting per-group CTA to RTA — AVOIDED]: g1/g2 kept as two KernelSpecs with per-group `N` CTA.
- **hw_config**: Style A — op resolves a TTNN `ComputeKernelConfig`; use `to_compute_hardware_config(device->arch(), compute_kernel_config)`. `unpack_modes`: legacy `unpack_to_dest_mode` was all-`Default` (→ `UnpackToSrc`); **when `fp32_dest_acc_en` (enable_32_bit_dest), add explicit `UnpackToSrc` entries for every Float32 DFB the compute kernel consumes** (in0, mask, max_scaler, sum_scaler, exps, recip_sum_exps, max, x_minus_max, tmp). DM `hw_config` via `create_reader/writer_datamovement_config(device->arch())`.

## Deferred / Flagged
- The shared kernel rewrite requires both ops' matching factories to flip together; a norm General factory left on `create_descriptor` after its kernel trio is rewritten would break. Units are ported as pairs.
- `tensor_args_t` reference members (rule 14) — flagged for ops team, not touched (report).
- **W-large unit ported, with a `noinline` workaround.** The Metal 2.0 port of `moreh_softmax_w_large.cpp` compiled on the host but initially failed to JIT-compile in the `fp32_dest_acc_en` path — an out-of-scope LLK addrmod `impossible constraint in 'asm'` cliff (confirmed a port regression vs legacy). Worked around by splitting the step-3 loop into a `static __attribute__((noinline))` helper so `kernel_main` stays under the compiler's constant-folding budget; behavior unchanged. See `METAL2_PORT_REPORT.md` → Handoff points. All five units (W-small, W-large, H-small, H-large, C-large) are ported and pass on device; the durable fix belongs upstream in the LLK.
