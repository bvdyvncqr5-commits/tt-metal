# Metal 2.0 Port Report — `normalization/softmax`

## Outcome

**PORTED (5 of 7 factories) + 2 remaining.** Five **General** factories
(`SoftmaxProgramFactoryGeneral{WSmall,HSmall,HLarge,CLarge,WLarge}`) are converted to `MetalV2FactoryConcept`
(`create_program_artifacts`), co-ported with `moreh/moreh_softmax` as the shared-kernel port-together set
(issue #51081), and pass on device.

`SoftmaxProgramFactoryGeneralWLarge` borrows the shared `moreh_softmax_w_large.cpp` compute kernel, whose
faithful Metal 2.0 port initially triggered an out-of-scope **LLK addrmod `impossible constraint in 'asm'`
JIT failure in the `fp32_dest_acc_en` path** (confirmed a port regression — legacy compiles the same test).
It is now **worked around** by a `noinline` split in that shared kernel (behavior-preserving stopgap; the
proper fix is upstream in the LLK). Full detail in `moreh/moreh_softmax/METAL2_PORT_REPORT.md` →
Handoff points; owner of the durable fix is the LLK team.

The two **Attention** factories
(`SoftmaxProgramFactoryAttentionOptimized` interleaved, `SoftmaxShardedProgramFactoryAttentionOptimized`
sharded) remain on the legacy `descriptor` concept and are enumerated as the next pass — see
[Open items](#open-items-for-downstream). The mixed-concept `program_factory_t` variant builds and
dispatches per-factory: ported General factories → `ProgramSpecFactoryConcept`, attention factories →
`ProgramDescriptorFactoryConcept`; `AllFactoriesValid` accepts the mix. Host build clean (exit 0).

This is a valid incremental deliverable per the recipe's factory-at-a-time guidance ("stopping after
[some factories] with the rest enumerated for the next pass is the expected shape for a large op"). The
General factories were the coordination-critical half (they force the shared-kernel rewrite that
`moreh/moreh_softmax` also depends on); the Attention factories are independent (own their kernels) and
can merge on their own timeline.

## Provenance

- **Recipe docs (this port):** `8cd19d4a006 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `8cd19d4a006 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for the five General factories (each returns `ProgramArtifacts{spec, run_params}`,
no op-owned tensors). Attention factories unchanged (`create_descriptor`).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none**.
- Pybind entry points removed: **none**.
- Header change (forced): `softmax_device_operation.hpp` — the five General factory structs now declare
  `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts` (added
  `#include <ttnn/metal_v2_artifacts.hpp>`). No edits to `validate_on_program_cache_miss`,
  `compute_output_specs`, `select_program_factory`, or the op's public `softmax(...)` entry points.

### Open items
- The two unreachable `return` statements in `select_program_factory` (device_operation.cpp:140,154, dead
  code after `std::visit`) were **left untouched** (off-limits op-level host code); flagged for the ops team.

## Handoff points

- **Cross-op shared kernels (co-migration, in this PR).** The five General factories borrow all their
  kernels by file path from `moreh/moreh_softmax/device/kernels/` (the 15-file set). Those kernels were
  rewritten to Metal 2.0 **in place**, and `moreh/moreh_softmax`'s five matching factories were flipped to
  `create_program_artifacts` in the same change (port-together set, issue #51081). See
  `moreh/moreh_softmax/METAL2_PORT_REPORT.md`. Both ops adopt the rewrite together; neither builds correctly
  against the rewritten kernels while still on `create_descriptor`.

## Successes

- **Mixed-concept `program_factory_t` variant** — porting 5 of 7 factories while leaving 2 on
  `create_descriptor` builds cleanly; `AllFactoriesValid` accepts the mix exactly as the recipe/TTNN-factory
  doc describe. This is what makes the incremental (factory-at-a-time) delivery real.
- Same successes as the moreh report (Same-FIFO alias, self-loop intermediates, preserved g1/g2
  multiplicity, `get_entry_size` on DM kernels) — the General factories share the kernels and pattern.

## Friction

### Gaps
- Same two as the moreh report: (1) DFB metadata getters have no DM form → used `get_entry_size()` on the
  DM reader/writer; (2) the "KernelRunArgs required for all kernels" vs "omit when no RTAs" ambiguity → no
  run-args entry for the RTA-less compute KernelSpecs (per recipe).

### Confusion
- **Norm vs moreh format asymmetry.** The General factories resolve intermediate/mask-scaler formats
  differently from moreh (`intermed = fp32?F32:F16_b`, `mask_scaler = bfp8?F16_b:df`) — but that lives in
  each factory's own `DataflowBufferSpec`, so the *shared* kernel is format-agnostic. No conflict; noting it
  because a reader diffing the two ops' factories will see intentional format divergence.

## Open items for downstream

- **Attention factories — remaining work (next pass).** Both are independent (own kernels under
  `device/kernels/attention/`), so they merge without the moreh coupling. Full inventory is in
  `METAL2_PORT_PLAN.md`. Notable complexity to budget for:
  - **Interleaved** (`SoftmaxProgramFactoryAttentionOptimized`, `softmax_program_factory_attention_optimized.cpp`):
    runtime source selection small/large (5 kernels convert atomically: `reader_unary_interleaved_sm{,_large_tensor}.cpp`,
    `writer_unary_interleaved_start_id_blocked_sm.cpp`, `compute/softmax{,_large_tensor}.cpp`). Conditional
    CBs across `FUSED_SCALE_MASK`/`CAUSAL_MASK`/`NUMERIC_STABLE`/`mask_padded_data`/`use_large_kernel`.
    Requires: promote host-known `mask_padded_data` to a `#define` to conditionally bind c_5 pad-mask
    (Conditional-DFB pattern); `#ifdef`-gate c_10 `cb_x` and the FUSED CBs (c_3/c_4/c_9); handle the
    no-mask `cb_x = cb_exps` Same-FIFO alias; `mask` is an optional Case-1 tensor binding.
  - **Sharded** (`SoftmaxShardedProgramFactoryAttentionOptimized`, `..._sharded.cpp`): borrowed-memory DFBs
    (`borrowed_from`) for c_0 input, c_11 output, and c_3 when the mask is sharded; Case-1 mask when
    interleaved/row-major; 3 runtime-selected readers (`reader_unary_sharded_sm{,_causal_mask_hw_dims,_rm_mask}.cpp`);
    numeric_stable conditional CBs (c_9/c_10); no writer.
  Neither was device-validatable in this pass's environment beyond host build; recommend porting each with
  the confirmed nightly `fused/` tests (`test_softmax_interleaved.py`, `test_softmax_sharded.py`).
- **Cross-op kernel touches:** the 15 borrowed kernels were modified **in place** (not forked); consumer set
  {`normalization/softmax` General, `moreh/moreh_softmax`} both migrated in this PR. Nothing to sunset.

## Verification

- **Build:** `./build_metal.sh --build-tests` → SUCCESS (exit 0, no errors, `AllFactoriesValid` satisfied).
- **Device tests:** `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/fused/test_softmax.py`
  → **373 passed, 1 skipped, 0 failed** (covers the ported General path — 3D/5D/multi-dim softmax — plus the
  untouched attention path; no JIT failures). The moreh nightly test (which shares the ported kernels,
  including w_large with the `noinline` workaround) is fully green (93 passed). The w_large+fp32 JIT failure
  surfaced only in that moreh nightly test — the fused suite here does not exercise General w_large + fp32 —
  and is now resolved by the shared-kernel workaround described above.
