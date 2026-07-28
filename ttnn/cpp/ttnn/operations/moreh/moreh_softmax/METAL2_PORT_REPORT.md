# Metal 2.0 Port Report — `moreh/moreh_softmax`

## Outcome

**PORTED (5 of 5).** All five factories — `MorehSoftmax{WSmall,HSmall,HLarge,CLarge,WLarge}Factory` —
converted to `MetalV2FactoryConcept` (`create_program_artifacts`) and pass on device. `MorehSoftmaxWLargeFactory`
initially failed to JIT-compile in the `fp32_dest_acc_en` path — an out-of-scope **LLK addrmod
`impossible constraint in 'asm'` cliff** surfaced by the port (legacy compiles the same test, so it is a
port-surfaced LLK/compiler cliff, not pre-existing). It is now **worked around** by a `noinline` split in the
shared `moreh_softmax_w_large.cpp` compute kernel that shrinks `kernel_main` back under the compiler's
constant-folding budget (behavior-preserving stopgap; the proper fix is upstream in the LLK). See
[Handoff points](#handoff-points).

This op is the **owner** half of the shared-kernel port-together set (issue #51081); all five units
were co-ported in one change with `normalization/softmax`'s matching General factories (which borrow the
same kernels). The mixed-concept `program_factory_t` variant builds and dispatches per-factory
(`AllFactoriesValid` accepts ported + legacy). Host build clean (`./build_metal.sh --build-tests`, exit 0).
Device tests: see [Verification](#verification).

## Provenance

- **Recipe docs (this port):** `8cd19d4a006 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `8cd19d4a006 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for all 5 factories — each returns `ProgramArtifacts{spec, run_params}` (no
op-owned tensors). The device-op `program_factory_t` variant now holds all-Metal-2.0 factories;
`AllFactoriesValid` passed (each satisfies exactly `ProgramSpecFactoryConcept`).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op had no custom hash).
- Pybind entry points removed: **none** (no pybind `create_descriptor`).
- Header change (forced, sanctioned): `moreh_softmax_device_operation.hpp` — the `DEFINE_SOFTMAX_FACTORY`
  macro now declares `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`
  (added `#include <ttnn/metal_v2_artifacts.hpp>`). This is the factory-signature flip the port forces,
  not a behavior edit to `validate`/`invoke`/`compute_output_specs`.

### Open items
- None on the concept fit — clean single-program port.

## Handoff points

- **RESOLVED (workaround) — `MorehSoftmaxWLargeFactory` / `moreh_softmax_w_large.cpp` compute kernel: LLK
  addrmod `impossible constraint in 'asm'` in the fp32 path.** The faithful Metal 2.0 port of the w_large
  compute kernel (CB-id → `dfb::` handles, `#include experimental/kernel_args.h`, named args — no logic
  change) failed to JIT-compile **only** when `fp32_dest_acc_en=True`, at
  `ckernel_addrmod.h:143` (`TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13))`)
  inlined from `reduce_init` via `reduce_helpers_compute.inl:362`, reached from the phase-1
  `compute_kernel_lib::reduce<MAX, REDUCE_ROW, ...>` and `mask_tile_to_cb` calls (moreh_softmax_w_large.cpp).
  `SETC16` is emitted via an inline-asm `"n"` (immediate) operand whose whole instruction word — the register
  address **and** the packed value (`dest.val() | (fidelity.val() << 13)`) — must fold to a compile-time
  constant; under the enlarged fp32 TU the compiler exceeds its constant-folding budget and can't. The addrmod
  header documents the same class of issue ("KCM - This gets around issue: error: impossible constraint in 'asm'").
  - **Confirmed a port regression, not pre-existing:** the identical test passes on legacy
    (`git stash` + rebuild + `pytest ...::test_softmax_large_algorithm_for_dim_hw` → 4 passed). The Metal 2.0
    kernel's marginally larger TU (generated binding headers + DFBAccessor construction) tips the compiler
    past the constant-folding cliff that legacy stayed under. **Only w_large+fp32 is affected** — w_small,
    h_small, h_large, c_large (incl. their fp32 variants) all pass.
  - **Workaround applied (in-scope, kernel-only):** the step-3 final-result loop was split into a
    `static __attribute__((noinline))` helper (`compute_final_result`), shrinking `kernel_main` back under the
    constant-folding budget so both `SETC16` immediates fold again. Behavior is unchanged (a plain function
    boundary). This is the single shared w_large compute kernel, so the one edit fixes both `moreh` WLarge and
    `normalization/softmax` GeneralWLarge on WH and BH. Verified on device (see [Verification](#verification)).
  - **Proper fix is upstream (out of porter scope):** the durable fix lives in
    `tt_metal/tt-llk/.../ckernel_addrmod.h` — make the addrmod value a compile-time constant so the immediate
    folds regardless of TU size. Templatizing only the section index (`set<mod_index>()`) was **verified
    insufficient**: it constant-folds the register address but not the packed value operand, so the failure
    persists. Owner: LLK team. Once fixed, the `noinline` stopgap can be removed.
- **Cross-op shared kernels (co-migration, in this PR).** 12 of the 15 kernels in `device/kernels/`
  (the `{reader,writer,}moreh_softmax_{w,h,h_large,c_large}.cpp` trios) were rewritten CB→DFB / named-token
  **in place** across both ops in this PR (the port-together set of issue #51081). The matching
  `normalization/softmax` General factories were flipped to `create_program_artifacts` in the same change.
  The 3 `*_w_large` kernels are ported too (with the `noinline` workaround in the compute kernel; see the
  resolution above), so both ops' w_large factories are `create_program_artifacts`. Consumers verified via
  `grep -rl <kernel> ttnn/cpp/ttnn/operations/` = {moreh_softmax, normalization/softmax}.
  `moreh_softmax_backward` is **not** a consumer (own kernels) — untouched.
- **`tensor_args_t` holds references (CLAUDE.md rule 14).** `struct tensor_args_t { const Tensor& input;
  const std::optional<Tensor>& output; }` (device op hpp) uses reference members, which rule 14 forbids.
  **Left as-is** (op-level host code, outside the factory-body port scope). Flagged for the moreh ops team
  as a latent structural cleanup; the readiness sheet already cleared the op `Is safe to port = yes`.

## Successes

- **Same-FIFO aliasing worked exactly as documented.** The compute kernels use the CB id in two spellings
  — as an object (`DataflowBuffer dfb_in0_obj(cb_in0)` for FIFO / `_with_dt` helpers) and as a raw
  `uint32_t` at LLK / `compute_kernel_lib::reduce<..., cb_in0, ...>` NTTP sites. `constexpr auto cb_in0 =
  dfb::in0;` + `DataflowBuffer dfb_in0_obj(cb_in0);` gives both from one binding, per
  [Same-FIFO aliasing](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md).
  The `DFBAccessor → uint32_t` implicit conversion flows into NTTP positions (`reduce<... cb_max ...>`) as
  the catalog promised.
- **Self-loop for compute-internal intermediates** (`exps`/`recip_sum_exps`/`max`/`x_minus_max`/`add`/`tmp`)
  bound PRODUCER+CONSUMER on the compute KernelSpec — the validator's ≥1P/≥1C rule is satisfied with no
  kernel change.
- **Preserved g1/g2 multiplicity** — two compute KernelSpecs of the same source over disjoint core groups,
  each with its own per-group CTA (`N`), placed in `wu_g1`/`wu_g2` (reader/writer span both). Avoided the
  [demote-CTA anti-pattern].

## Friction

### Gaps
- **DFB metadata getters have no DM form (rule 7).** The DM reader/writer kernels used
  `get_tile_size(cb_id)` for the NoC transfer byte count. Rule 7 says rewrite to a member getter, but
  `DataflowBuffer::get_tile_size()` is gated behind `DFB_DESCRIPTORS_DEFINED` (compute-only). The DM-safe
  equivalent is `dfb.get_entry_size()` (whitelist §B, always available; equals the tile byte size here).
  Used that. Worth a one-line note in the recipe's rule 7 that DM kernels take `get_entry_size()`.
- **"A `KernelRunArgs` must be specified for ALL kernels" vs "omit when no RTAs."** `program_run_args.hpp`
  says a `KernelRunArgs` is required for every kernel; the recipe says the run-args entry may be omitted for
  a kernel with no RTAs. The compute KernelSpecs have only CTAs (no RTA schema), so no `KernelRunArgs`
  entries were emitted for them. Followed the recipe; pending device-test confirmation this is accepted by
  the validator.

### Confusion
- None material — the moreh/General factories are near-clones and the pattern replicated cleanly across all
  5 units once the first (W-small) built.

## Open items for downstream

- **Cross-op kernel touches:** the 15 `device/kernels/*.cpp` were modified **in place** (not forked) — the
  bundled consumer set is {`moreh/moreh_softmax`, `normalization/softmax` General factories}, both migrated
  in this PR. No `_metal2` fork; no remaining unmigrated consumer. Nothing to sunset.
- **`unpack_modes` thoroughness:** for the fp32 path (`enable_32_bit_dest`) an explicit `UnpackToSrc` entry
  was added for **every** Float32 DFB the compute kernel consumes (matching the legacy all-`Default`
  vector). If the validator flags an entry for a DFB it deems un-consumed, trim per its message — the set
  is intentionally inclusive.
- Attention factories are in `normalization/softmax`, not this op; see that op's report.

## Verification

- **Build:** `./build_metal.sh --build-tests` → SUCCESS (exit 0, 0 `error:`, `AllFactoriesValid` satisfied).
- **Device tests:** `scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py`
  - Before the `noinline` workaround (all 5 ported): **92 passed, 1 failed, 32 skipped** — the single failure
    was the w_large+fp32 JIT build error described above.
  - **After the `noinline` workaround (all 5 ported): 93 passed, 32 skipped, 0 failed.** Every case
    (w_small / h_small / h_large / **w_large**, softmax / softmin / logsoftmax, bf16 / bfp8 / fp32) passes
    with pcc ≈ 0.9999. `pytest ...::test_softmax_large_algorithm_for_dim_hw` → 4 passed (the previously failing
    w_large+fp32 combo).
  - Also relevant: `test_moreh_logsoftmax.py`, `test_moreh_logsoftmax_ulp.py` (LOG define path).
- **Legacy baseline (that established the port regression):** `git stash` + rebuild +
  `pytest ...::test_softmax_large_algorithm_for_dim_hw` → **4 passed** — confirming w_large+fp32 was a port
  regression, not pre-existing.
