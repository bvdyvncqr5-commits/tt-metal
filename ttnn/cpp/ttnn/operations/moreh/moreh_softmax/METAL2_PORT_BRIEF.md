# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_softmax`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `8cd19d4a006 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

**Scope:** one DeviceOperation (`MorehSoftmaxOperation`), five interleaved factories — WSmall / WLarge / HSmall / HLarge / CLarge (`device/softmax_*/`).

> **Port-together dependency:** this op **owns** the 15 kernels in `device/kernels/` that `normalization/softmax`'s five General factories borrow by file path. The kernel-side rewrite is shared — port this op **together with** `normalization/softmax`'s General factories (tracked in issue #51081), or the co-borrower breaks. Do not migrate the shared kernels for one op alone.

## TTNN factory analysis

- **Current concept:** `descriptor` (all 5 factories).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (plain).
- **Gate-cleared, confirmed absent:** custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no`; `Is safe to port? = yes`, `Smuggled pointer = no`.

## Construct — to do

**Tensor bindings** (per binding; all factories interleaved):

- `input` — **Case 1** → express as `TensorParameter` / `TensorBinding`; reader builds `TensorAccessor(tensor::name)` in place of `TensorAccessor(in_args, src_addr)`. The `Buffer*`/`src_addr` RTA and its `TensorAccessorArgs` CTAs disappear.
- `output` — **Case 1** → same on the writer side (`TensorAccessor(out_args, dst_addr)` → `TensorAccessor(tensor::name)`).
- No Case 2; no borrowed-memory DFB.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every `TensorAccessor` is already 2-arg.

**CB endpoints** (compute `g1`/`g2` cover **disjoint** core groups → one compute instance per node; ordinary 1:1):

- **Assign 1P+1C:** `c_0` input (R→C), `c_1` mask (R→C), `c_2` max-scaler (R→C), `c_3` sum-scaler (R→C), `c_16` output (C→W).
- **Self-loop** the compute-internal intermediates: `c_24` exp, `c_25` reduce, `c_26` max, `c_27` x-max, `c_28` tmp.
- No multi-binding flag; no dead-CB drop.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** the 15 `device/kernels/*.cpp` files are shared with `normalization/softmax` (General factories) — **one kernel rewrite, both ops adopt together** (issue #51081). `moreh/moreh_softmax_backward` is not coupled.
- **RTA varargs:** none — name every RTA (all read at fixed constexpr indices).
