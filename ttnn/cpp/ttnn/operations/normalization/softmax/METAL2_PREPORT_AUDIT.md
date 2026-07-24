# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/normalization/softmax`

One DeviceOperation, seven ProgramFactory variants:

- **`SoftmaxDeviceOperation`** (`device/softmax_device_operation.hpp`)
  - `SoftmaxProgramFactoryGeneralWSmall` (`softmax_program_factory_general_w_small.cpp`)
  - `SoftmaxProgramFactoryGeneralWLarge` (`softmax_program_factory_general_w_large.cpp`)
  - `SoftmaxProgramFactoryGeneralHSmall` (`softmax_program_factory_general_h_small.cpp`)
  - `SoftmaxProgramFactoryGeneralHLarge` (`softmax_program_factory_general_h_large.cpp`)
  - `SoftmaxProgramFactoryGeneralCLarge` (`softmax_program_factory_general_c_large.cpp`)
  - `SoftmaxProgramFactoryAttentionOptimized` (`softmax_program_factory_attention_optimized.cpp`) — interleaved
  - `SoftmaxShardedProgramFactoryAttentionOptimized` (`softmax_program_factory_attention_optimized_sharded.cpp`) — sharded

The five **General** factories file-path-instantiate their kernels from `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/` (a cross-family borrow — see Team-only). The two **Attention** factories instantiate the op's own kernels under `device/kernels/attention/`. All kernel files under `device/kernels/attention/` are referenced by one of the two attention factories (no unreferenced/dead kernel files).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `8cd19d4a006 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/normalization/softmax` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `SoftmaxDeviceOperation` → 5 General (WSmall/WLarge/HSmall/HLarge/CLarge) + 2 Attention (interleaved, sharded) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 24 referenced kernels are Device 2.0 (own attention kernels + moreh_softmax donor kernels) |
| *Prereqs* — Cross-op escapes | Ok — all `#include` escapes resolve to shared pools (`kernel_lib`, `kernel/`, `kernel_helper_functions`) + tt_metal `api/`; no cross-family function-call escape. File-path borrow of the moreh_softmax kernels induces a **port-together set with `moreh/moreh_softmax`** (FYI, non-gating; `moreh/moreh_softmax` is unaudited). |
| *Feature Support* — overall | **GREEN** (every Appendix A entry N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (all 7 factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 7) |
| *TTNN Readiness* — Secretly SPMD | N/A (`descriptor`, not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes (all 7) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (no address fold on any factory) |
| *Port work* — Tensor bindings (per binding) | Case 1 (interleaved/general I/O + interleaved mask) · clean/borrowed-DFB (sharded I/O + sharded mask) — no Case 2 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (every `TensorAccessor` is 2-arg) |
| *Port work* — CB endpoints | legal / self-loop / 1P+1C (no multi-binding, no dead CB) |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves to a **self-loop** (compute-internal intermediates; single-toucher borrowed-memory CBs) or a plain **1P+1C** (reader→compute→writer FIFO pairs). No CB needs the multi-binding flag; no dead CB was confirmed. Dispositions are recorded per `(CB, config)` below.

## Result

**GREEN → brief issued.** All five gates clear for all seven factories:
Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓.
The port can begin (after explicit user go-ahead). Port work is routine: express tensor bindings (Case 1 / borrowed-DFB), self-loop / 1P+1C the CBs, and coordinate the shared moreh_softmax kernel rewrite as a port-together set with `moreh/moreh_softmax`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet (*"Operations analysis"*, fetched this run) carries all seven `normalization/softmax` / `SoftmaxDeviceOperation` factory rows with `Is able to port? = yes`. Every conjunct clears: `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Is safe to port? = yes`, `Smuggled pointer = no`, `TensorParameter relaxation = none`. Cross-check against the code confirms each cheaply-checkable column:
  - `Concept = descriptor` ✓ — each factory is a `static ProgramDescriptor create_descriptor(...)` (device/softmax_device_operation.hpp:25-57).
  - `Custom hash = no` ✓ — no `compute_program_hash` override anywhere in the op directory (grep: none).
  - `Runtime-args update = no` ✓ — no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep: none).
  - `Pybind descriptor = no` ✓ — no `create_descriptor` binding in `softmax_nanobind.cpp` (grep: none).
  No cross-column invariant is violated (no op-owned tensors on a `descriptor` row). Sheet and code agree.

- **Device 2.0 (every kernel used):** **GREEN.** Every one of the 24 referenced kernels — the 15 borrowed moreh_softmax kernels and the 9 own attention kernels — is structurally Device 2.0. Confirmed by (a) a repo-wide idiom scan finding **zero** Device-1.0 signatures (`InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen` / `get_noc_addr_from_bank_id` / raw `noc_async_read(` / `noc_async_write(` / `noc_semaphore_*`), **zero** free-function FIFO ops (`cb_reserve_back(` / `cb_push_back(` / `cb_wait_front(` / `cb_pop_front(`), and **zero** free-function `get_read_ptr`/`get_write_ptr` holdovers; and (b) reading representative kernels of every role. The kernels use the Device-2.0 / DFB object model throughout: `Noc`, `DataflowBuffer` / `CircularBuffer` wrapper objects, `TensorAccessor`, object-form `.reserve_back()/.push_back()/.wait_front()/.pop_front()`, `noc.async_read()/async_write()` + barriers, and `tile_regs_acquire/commit/wait/release` in compute. The only CB-index free function present is `get_tile_size(cb_id)`, which is **sanctioned** (kept by Device 2.0). Raw pointer writes exist (`cb_*_obj.get_write_ptr()` for scaler/mask/pad generation) but are all **object-form** method calls by the CB's own FIFO producer, paired with `push_back` — not holdovers and not hidden second writers. No `evil_set_*_ptr` cursor mutation.

  Kernels verified in full: `reader_moreh_softmax_w.cpp`, `moreh_softmax_w.cpp` (moreh donors); `reader_unary_interleaved_sm.cpp`, `writer_unary_interleaved_start_id_blocked_sm.cpp`, `reader_unary_sharded_sm.cpp`, `compute/softmax.cpp` (attention). Remaining kernels confirmed by the negative idiom scan + include inventory.

- **Feature compatibility:** every Appendix A entry is N/A — a clean all-`N/A` scan.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `.global_circular_buffer` / `CreateGlobalCircularBuffer` / `remote_index` / `remote_cb` anywhere (grep: none). The sharded factory's `.buffer = <buffer>` fields (c_0/c_3/c_11) are the ordinary borrowed-memory pattern, not GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` / `set_address_offset` field set anywhere (grep: none). |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `global_semaphore` (grep: none). No semaphores at all in this op. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | No runtime-varying CTA index. `tensor_args_t` is a fixed pair (`input_tensor`, optional `mask`) — no `std::vector<Tensor>`. The `get_compile_time_arg_val(mask_args.next_compile_time_args_offset() + N)` reads (attention readers) are `constexpr` computed offsets, not runtime-varying loop indices — fixed-count per the entry's false-positive guard. |

- **CB endpoints (GATE-free):** every CB resolves to a port-time disposition; nothing blocks. See the Port-work summary for the per-`(CB, config)` inventory. No multi-binding flag is needed and no dead CB was confirmed.

- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into its base on **any** factory. There is **no `->address()` call anywhere** in the seven factories — every tensor base reaches the kernel as a `Buffer*` pushed into `emplace_runtime_args` (the framework `BufferBinding` form), and every offset (`tile_offset`, `mask_id`, `mask_start_tile_id`, `mask_offset`, plus the c_large strides `outer_stride`/`inner_size`/`dim_size`) is passed as a **separate scalar** and applied kernel-side as a `page_id`. Consistent with the offset-base-pointer triage analysis (dated 2026-07-19), which does not list softmax.

- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor` construction across all kernels is 2-arg (`TensorAccessor(args, addr)`) — no explicit page-size third argument at any of the 19 construction sites. Consistent with the 3rd-arg triage analysis (dated 2026-07-06), which does not list softmax.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory):
  - **General factories** (WSmall/WLarge/HSmall/HLarge/CLarge): `input` → **Case 1** (reader builds `TensorAccessor(in_args, src_addr)`); `output` → **Case 1** (writer builds `TensorAccessor(out_args, dst_addr)`). Base delivered as `Buffer*` RTA today.
  - **Attention interleaved** (`SoftmaxProgramFactoryAttentionOptimized`): `input` → **Case 1**; `output` → **Case 1**; `mask` (optional) → **Case 1** (`TensorAccessor(mask_args, mask_addr)`).
  - **Attention sharded** (`SoftmaxShardedProgramFactoryAttentionOptimized`): `input` → **clean** (borrowed-memory DFB, c_0 `.buffer = src0_buffer`); `output` → **clean** (borrowed-memory DFB, c_11 `.buffer = out0_buffer`); `mask` when **sharded** → **clean** (borrowed-memory DFB, c_3 `.buffer = mask_buffer`); `mask` when **interleaved/row-major** → **Case 1** (`TensorAccessor(mask_args, mask_addr)`). Per-binding classification of `mask` thus varies by config — record the split in the port.
  - **No Case 2** anywhere: every non-borrowed tensor base flows through a `TensorAccessor`; no kernel does raw base-pointer arithmetic.
- **TensorParameter relaxation:** none (sheet `none`, no custom hash to reconcile).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (disposition per `(CB, config)`; compute instances `g1`/`g2` in the General factories cover **disjoint** core groups, so each node sees one compute instance — ordinary 1:1, not a dual-instance work-split):
  - **General factories** (reader R + writer W + compute C): `c_0` in R→C **1P+1C**; `c_1` mask R→C **1P+1C**; `c_2` max-scaler R→C **1P+1C**; `c_3` sum-scaler R→C **1P+1C**; `c_16` out C→W **1P+1C**; `c_24` exp / `c_25` reduce / `c_26` max / `c_27` x-max / `c_28` tmp — compute-internal, single toucher → **self-loop**.
  - **Attention interleaved** (R + W + C): `c_0` in0 R→C **1P+1C**; `c_2` max-scaler R→C **1P+1C**; `c_13` sum-scaler R→C **1P+1C**; `c_3` fused-scale R→C **1P+1C** (FUSED_SCALE_MASK); `c_4` fused-attn/mask R→C **1P+1C** (FUSED_SCALE_MASK); `c_5` pad-mask W→C **1P+1C** (only when `mask_padded_data`; **porter: census per-config** — untouched when padding absent, resolve as single-ended/unused then); `c_11` out0 C→W **1P+1C**; `c_6` exps / `c_7` recipsumexps / `c_9` scale-mask / `c_8` max (NUMERIC_STABLE) / `c_10` cb_x (NUMERIC_STABLE) / `c_12` `c_15` `c_16` (large-kernel intermeds) — compute-internal → **self-loop**.
  - **Attention sharded** (R + C, no writer): `c_0` input (borrowed) compute-only → **self-loop**; `c_11` output (borrowed) compute-only → **self-loop**; `c_3` mask — **self-loop** when sharded-borrowed (compute-only), **1P+1C** (R→C) when interleaved; `c_1` max-scaler R→C **1P+1C**; `c_13` sum-scaler R→C **1P+1C**; `c_2` fused-scale R→C **1P+1C**; `c_6` `c_7` `c_8` `c_9` `c_10` intermeds compute-only → **self-loop**.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader, no dual-instance co-fill. The raw `get_write_ptr()` writes (reader scaler/mask generation; writer pad-mask generation) are single-producer FIFO fills, not co-fills. The only per-config care item is `c_5` (pad-mask) in the attention interleaved factory — census it per `mask_padded_data`.
- **Cross-op / shared kernels:** the five General factories borrow **all** their kernels (reader/writer/compute × W/W-large/H/H-large/C-large = 15 files) by file path from `moreh/moreh_softmax/device/kernels/` — which holds exactly those 15 files and no others, i.e. it *is* moreh_softmax's entire kernel set. Those same files are co-instantiated by `moreh/moreh_softmax` (`MorehSoftmaxOperation`). Their Metal 2.0 kernel-side rewrite is a **single change** that must land across the whole **port-together set = {normalization/softmax general factories, moreh/moreh_softmax}** — see Team-only. (`moreh/moreh_softmax_backward` is **not** in the set — it uses its own `moreh_softmax_backward/device/kernels/`, never the forward path.)
- **RTA varargs:** none — every kernel reads RTAs at fixed constexpr indices as distinct named fields (no `arg_index++` loop, no data-selected index, no `get_common_arg_val`).

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Roll-up: ✓ clean** for function-call escapes. Every cross-directory `#include` resolves to a shared pool, and every borrowed function's signature is a Device-2.0-native shape:
    - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_{dataflow,compute}.hpp` (official kernel library, lib team) — `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_id, ...>()`, `compute_kernel_lib::reduce<..., cb_id, ...>()`: CB indices as `uint32_t` NTTPs → ✓ (`dfb::name` constexpr cast).
    - `ttnn/cpp/ttnn/kernel/{compute,dataflow}/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` (singular `kernel/` shared pool, treat as shared-lib) — `mask_tile_to_cb(DataflowBuffer&, ...)`, `generate_mask_w<T>(DataflowBuffer&, ...)`, `generate_bcast_unary_scalar(CircularBuffer, ...)`: DFB/CB objects → ✓ excellent (Device-2.0 native).
    - `ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp` (shared utility pool) — `fill_pad_tile<T, N, M>(uint32_t ptr, value)`: raw L1 pointer + scalar, no resource handle → ✓ (no binding token needed).
    - Host-only: General factories `#include "ttnn/operations/moreh/moreh_helper_functions.hpp"` for `split_work_to_cores_wt_core_range` — host helper, not a kernel; no bearing on the kernel-token translation.
  - **Borrowed kernel files (file-path instantiation):** the 15 `moreh/moreh_softmax/device/kernels/*.cpp` files listed below are instantiated by the General factories but **owned by the moreh_softmax family**. That directory contains **exactly** these 15 files (verified) — it is moreh_softmax's whole kernel set, so moreh_softmax is entirely kernel-coupled to the General path. **Port-together set = {`normalization/softmax` General factories, `moreh/moreh_softmax`}** (the shared kernel-side rewrite must adopt across both in one change, or migrate the shared kernels as their own unit first):
    - `reader_moreh_softmax_{w,w_large,h,h_large,c_large}.cpp`, `writer_moreh_softmax_{w,w_large,h,h_large,c_large}.cpp`, `moreh_softmax_{w,w_large,h,h_large,c_large}.cpp` — shared by **`normalization/softmax`** (General factories) **and `moreh/moreh_softmax`** (`MorehSoftmaxOperation`, factories WSmall/WLarge/HSmall/HLarge/CLarge).
    - **`moreh/moreh_softmax_backward` is NOT in the set.** It uses only its own `moreh_softmax_backward/device/kernels/` (its own `reader_moreh_softmax_backward_*` and even its own same-named `writer_moreh_softmax_{w,h}.cpp` copies) — it never instantiates the forward `moreh_softmax/device/kernels/` path. (An earlier substring match on `reader_moreh_softmax` suggested otherwise; corrected here.)
    - **`moreh/moreh_softmax` itself has not been audited** — clearing the General factories for merge requires a separate Metal 2.0 audit of `moreh/moreh_softmax` (its own host factories, bindings, offset-pointers, CB endpoints), then a coordinated co-port. The shared kernels being Device 2.0 does not clear moreh_softmax's host side.
  - The attention factories own their kernels (`device/kernels/attention/`) — no borrow; portable independently of moreh_softmax.
- **Relaxation candidates:** none — no custom hash to mine.
- **TTNN factory analysis:** current concept `descriptor` (all 7); no op-owned tensors (no `WorkloadDescriptor`, no `buffers` vector); no custom hash; no custom `override_runtime_arguments`; no pybind `create_descriptor`; `Is safe to port? = yes` / `Smuggled pointer = no` (readiness-sheet owner's correctness call). Target concept `MetalV2FactoryConcept` for every factory (feeds the port's TTNN ProgramFactory wiring per `ttnn_factory.md`).

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Unreachable `return` statements** in `select_program_factory` — `device/softmax_device_operation.cpp:140` and `:154` each sit immediately after a `std::visit(...)` that already returns in every branch, so they are dead code. Harmless; route to the ops team for cleanup.
- **Kernels created on unused cores** — the attention interleaved factory sets `core_ranges = all_device_cores` for reader/writer/compute and hands unused cores (`i >= num_cores`) all-zero runtime args (`softmax_program_factory_attention_optimized.cpp:396-405`) rather than restricting kernels to `all_cores`. Existing behavior, not a port concern; noting as a latent inefficiency against CLAUDE.md's "don't create kernels on unused cores" convention.

## Recipe notes

- The audit's TensorParameter-analysis detection shapes are written around `->address()`-in-RTA and explicit `Buffer*`-in-RTA. This op uses the `Buffer*`-binding form uniformly (`emplace_runtime_args(core, {input_tensor.buffer(), ...})`) with **no** `->address()` call anywhere, and the recipe handles this cleanly (classify by kernel use → Case 1). Worked as written; no friction — noting only that a reader expecting to grep `->address()` will find zero hits here and should fall through to the `Buffer*`-binding shape.
