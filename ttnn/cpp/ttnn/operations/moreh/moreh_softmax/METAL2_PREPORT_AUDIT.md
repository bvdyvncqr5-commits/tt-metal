# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_softmax`

One DeviceOperation, five ProgramFactory variants:

- **`MorehSoftmaxOperation`** (`device/moreh_softmax_device_operation.hpp`)
  - `MorehSoftmaxWSmallFactory` (`device/softmax_w_small/softmax_w_small.cpp`)
  - `MorehSoftmaxWLargeFactory` (`device/softmax_w_large/softmax_w_large.cpp`)
  - `MorehSoftmaxHSmallFactory` (`device/softmax_h_small/softmax_h_small.cpp`)
  - `MorehSoftmaxHLargeFactory` (`device/softmax_h_large/softmax_h_large.cpp`)
  - `MorehSoftmaxCLargeFactory` (`device/softmax_c_large/softmax_c_large.cpp`)

All five factories file-path-instantiate their kernels from this op's own `device/kernels/` directory, which holds **exactly 15 files** (reader/writer/compute × W/W-large/H/H-large/C-large). **These same 15 files are borrowed by `normalization/softmax`'s five General factories** — moreh_softmax is the *owner* side of that port-together coupling (see Team-only). `MorehSoftmaxOp` covers SOFTMAX / SOFTMIN / LOGSOFTMAX (compile-time defines on the same kernels). This op has **no sharded factory** — all five variants are interleaved.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`. Audited as the port-together partner of `normalization/softmax` (issue #51081); findings merged into that ticket rather than a separate sub-issue.

**Recipe docs:** `8cd19d4a006 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_softmax` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehSoftmaxOperation` → WSmall / WLarge / HSmall / HLarge / CLarge (all interleaved) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 15 referenced kernels are Device 2.0 (same shared set audited for `normalization/softmax`) |
| *Prereqs* — Cross-op escapes | Ok — `#include` escapes resolve to shared pools (`kernel_lib`, `kernel/`) + tt_metal `api/`; no cross-family function-call escape. Owns the 15 kernels borrowed by `normalization/softmax` (port-together set). |
| *Feature Support* — overall | **GREEN** (every Appendix A entry N/A) |
| *Feature Support* — Variadic-CTA | Ok — `tensor_args_t` is `{input, optional output}`, no `std::vector<Tensor>`; kernels read CTAs at constexpr offsets only |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (all 5 factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 5) |
| *TTNN Readiness* — Secretly SPMD | N/A (`descriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes (all 5) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (no address fold on any factory) |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `output` Case 1 — no Case 2, no borrowed-DFB (no sharded factory) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (every `TensorAccessor` is 2-arg) |
| *Port work* — CB endpoints | 1P+1C (I/O + scalers) / self-loop (compute-internal intermediates); no multi-binding, no dead CB |

## Result

**GREEN.** All five gates clear for all five factories: Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓. The op is structurally a near-clone of `normalization/softmax`'s General path (the General factories were derived from these). **Because the two ops share the same 15 kernel `.cpp` files as a single source of truth, they form a port-together set** — the Metal 2.0 kernel-side rewrite (CB/DFB index → named `dfb::`/`tensor::` tokens) must land in `moreh/moreh_softmax` and `normalization/softmax`'s General factories in one coordinated change.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet carries all five `moreh/moreh_softmax` / `MorehSoftmaxOperation` factory rows with `Is able to port? = yes`; every conjunct clears (`Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Is safe to port? = yes`, `Smuggled pointer = no`, `TensorParameter relaxation = none`). Code cross-check confirms: each factory is a `static ProgramDescriptor create_descriptor(...)` (device op hpp:49-62); no `compute_program_hash` (grep: none); no `get_dynamic_runtime_args`/`override_runtime_arguments` (grep: none); no `create_descriptor` binding in `moreh_softmax_nanobind.cpp` (grep: none). No cross-column invariant violated. Sheet and code agree.
- **Device 2.0 (every kernel used):** **GREEN.** All five factories instantiate only the 15 kernels in `moreh/moreh_softmax/device/kernels/` — the identical set already verified Device-2.0 compliant during the `normalization/softmax` audit (DFB/`CircularBuffer` object model, `Noc`, `TensorAccessor`, object-form FIFO; only the sanctioned `get_tile_size(cb_id)` free function; no Device-1.0 idioms, no free-function FIFO/ptr holdovers, no `evil_set_*_ptr`).
- **Feature compatibility:** every Appendix A entry N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | none (grep: no `GlobalCircularBuffer` / `remote_index` / `.global_circular_buffer`) |
  | CBDescriptor `address_offset` (non-zero) | N/A | none (grep: no `address_offset`) |
  | GlobalSemaphore | N/A | none; no semaphores in this op |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` = `{const Tensor& input, const std::optional<Tensor>& output}` — fixed count; kernels read CTAs at constexpr offsets only |

- **Offset base pointers:** **GREEN.** No `->address()` call in any factory; every base reaches the kernel as a `Buffer*` in `emplace_runtime_args`, offsets (`tile_offset`, and c_large's `outer_stride`/`inner_size`/`dim_size`) passed as separate scalars → clean bases. Not listed in the offset-base-pointer triage (2026-07-19).
- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor` in the shared kernels is 2-arg. Not listed in the 3rd-arg triage (2026-07-06).

## Port-work summary

- **Tensor bindings** (per binding, all 5 factories, all interleaved):
  - `input` → **Case 1** (reader builds `TensorAccessor(in_args, src_addr)`; base delivered as `Buffer*` RTA today).
  - `output` → **Case 1** (writer builds `TensorAccessor(out_args, dst_addr)`).
  - No Case 2 (no raw base-pointer arithmetic); no borrowed-memory DFB (no sharded factory).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (reader R + writer W + compute C; compute `g1`/`g2` cover **disjoint** core groups → one compute instance per node, ordinary 1:1):
  - `c_0` input R→C **1P+1C**; `c_1` mask R→C **1P+1C**; `c_2` max-scaler R→C **1P+1C**; `c_3` sum-scaler R→C **1P+1C**; `c_16` output C→W **1P+1C**.
  - `c_24` exp / `c_25` reduce / `c_26` max / `c_27` x-max / `c_28` tmp — compute-internal, single toucher → **self-loop**.
  - No multi-binding, no dead CB.

## Heads-ups

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** this op **owns** the 15 kernels in `device/kernels/` that `normalization/softmax`'s five General factories borrow by file path. Their Metal 2.0 kernel-side rewrite is one change that must land across **both** ops (port-together set) — see Team-only. `moreh/moreh_softmax_backward` is *not* in the set (it has its own `device/kernels/`).
- **RTA varargs:** none (all RTAs read at fixed constexpr indices as distinct fields).

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Roll-up: ✓ clean** for function-call escapes — kernel `#include`s resolve to `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_{dataflow,compute}.hpp` (kernel library; CB-index NTTPs → ✓) and `ttnn/cpp/ttnn/kernel/{compute,dataflow}/moreh_common.hpp`, `.../generate_bcast_scalar.hpp` (singular `kernel/` shared pool; DFB/CB object params → ✓ Device-2.0 native). Host-only: factories `#include "ttnn/operations/moreh/moreh_helper_functions.hpp"` for `split_work_to_cores_wt_core_range` — host helper, not a kernel.
  - **Owned kernel files (the port-together set):** `reader_moreh_softmax_{w,w_large,h,h_large,c_large}.cpp`, `writer_moreh_softmax_{w,w_large,h,h_large,c_large}.cpp`, `moreh_softmax_{w,w_large,h,h_large,c_large}.cpp` — owned here, **co-instantiated by `normalization/softmax`** (General factories WSmall/WLarge/HSmall/HLarge/CLarge). One rewrite, both ops adopt together. **`moreh/moreh_softmax_backward` is NOT coupled** — it uses only its own `moreh_softmax_backward/device/kernels/`.
- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:** current concept `descriptor` (all 5); no op-owned tensors; no custom hash; no custom `override_runtime_arguments`; no pybind `create_descriptor`; `Is safe to port? = yes` / `Smuggled pointer = no`. Target `MetalV2FactoryConcept` for every factory.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **`tensor_args_t` holds references** — `struct tensor_args_t { const Tensor& input; const std::optional<Tensor>& output; }` (device op hpp:41-44) uses reference members, which CLAUDE.md rule 14 ("`tensor_args_t` must NOT contain references") forbids. The readiness sheet marks the op `Is safe to port? = yes`, so the sheet owner has cleared it for porting; flagging for the ops team as a latent structural cleanup, not a port blocker.
