# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/normalization/softmax`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `8cd19d4a006 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

**Scope:** one DeviceOperation (`SoftmaxDeviceOperation`), seven factories — 5 General (WSmall/WLarge/HSmall/HLarge/CLarge, `device/softmax_program_factory_general_*.cpp`) + 2 Attention (interleaved `..._attention_optimized.cpp`, sharded `..._attention_optimized_sharded.cpp`).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); every factory ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all 7 factories — each is a `create_descriptor` returning `ProgramDescriptor`).
- **Op-owned tensors:** none (no `WorkloadDescriptor`, no `buffers` vector).
- **Target concept:** `MetalV2FactoryConcept` (plain — no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no`; `Is safe to port? = yes`, `Smuggled pointer = no`.

## Construct — to do

**Tensor bindings** (per binding; classification of `mask` varies by config in the sharded factory — carry the split):

- **General factories** (WSmall/WLarge/HSmall/HLarge/CLarge):
  - `input` — **Case 1** → express as `TensorParameter` / `TensorBinding`; reader builds `TensorAccessor(tensor::name)` in place of `TensorAccessor(in_args, src_addr)`. The `Buffer*`/`src_addr` RTA and its `TensorAccessorArgs` CTAs disappear.
  - `output` — **Case 1** → same, writer side (`TensorAccessor(out_args, dst_addr)` → `TensorAccessor(tensor::name)`).
- **Attention interleaved** (`SoftmaxProgramFactoryAttentionOptimized`):
  - `input` — **Case 1**; `output` — **Case 1**; `mask` (optional) — **Case 1** (`TensorAccessor(mask_args, mask_addr)` → `TensorAccessor(tensor::name)`).
- **Attention sharded** (`SoftmaxShardedProgramFactoryAttentionOptimized`):
  - `input` — **clean** (borrowed-memory DFB; c_0 today `.buffer = src0_buffer`) → port via `DataflowBufferSpec::borrowed_from`.
  - `output` — **clean** (borrowed-memory DFB; c_11 `.buffer = out0_buffer`) → `borrowed_from`.
  - `mask` **sharded** — **clean** (borrowed-memory DFB; c_3 `.buffer = mask_buffer`) → `borrowed_from`.
  - `mask` **interleaved / row-major** — **Case 1** (`TensorAccessor(mask_args, mask_addr)` → `TensorAccessor(tensor::name)`).
- **No Case 2** anywhere — no kernel does raw base-pointer arithmetic; every non-borrowed base flows through a `TensorAccessor`. No `get_bank_base_address` bridge is needed.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every `TensorAccessor` is already 2-arg; nothing to drop.

**CB endpoints** (compute `g1`/`g2` in the General factories cover **disjoint** core groups → each node sees one compute instance; ordinary 1:1, not a dual-instance work-split):

- **Self-loop** the compute-internal intermediates (single toucher): General `c_24/c_25/c_26/c_27/c_28`; attention interleaved `c_6/c_7/c_9/c_8/c_10` (+ large-kernel `c_12/c_15/c_16`); sharded `c_6/c_7/c_8/c_9/c_10`; and the sharded borrowed-memory CBs touched only by compute (`c_0` input, `c_11` output, `c_3` when sharded-mask).
- **Assign 1P+1C** the reader→compute and compute→writer FIFO pairs: General `c_0/c_1/c_2/c_3` (R→C) and `c_16` (C→W); attention interleaved `c_0/c_2/c_13/c_3/c_4` (R→C), `c_5` (W→C, only when `mask_padded_data`), `c_11` (C→W); sharded `c_1/c_13/c_2` (R→C) and `c_3` when interleaved-mask (R→C).
- **No multi-binding flag**; **no dead-CB drop**.
- **Census `c_5` (attention interleaved pad-mask) per config:** it is W→C 1P+1C when `mask_padded_data`, and untouched when padding is absent — resolve as single-ended/unused under the non-padded config rather than assuming a fixed disposition.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader, no dual-instance co-fill. The raw `get_write_ptr()` writes (reader scaler/mask generation, writer pad-mask generation) are single-producer FIFO fills paired with `push_back`, not co-fills. Bind each as the CB's PRODUCER; don't reach for the flag.
- **Cross-op / shared kernels:** the five General factories borrow **all** their kernels by file path from `moreh/moreh_softmax/device/kernels/` (exactly 15 files = moreh_softmax's whole kernel set). These same files are co-instantiated by `moreh/moreh_softmax` (`MorehSoftmaxOperation`). Their CB→DFB / named-token rewrite is **one change that must land across the whole port-together set = {normalization/softmax General factories, moreh/moreh_softmax}** — do not migrate the General factories' kernels in isolation, or moreh_softmax breaks. **`moreh/moreh_softmax` is unaudited**; it needs its own Metal 2.0 audit + a coordinated co-port before the General factories can merge. (`moreh/moreh_softmax_backward` is *not* coupled — it uses its own backward kernels.) **The two Attention factories own their kernels and can be ported freely, independent of moreh_softmax — port these first.**
- **RTA varargs:** none — name every RTA (all read at fixed constexpr indices as distinct fields).
