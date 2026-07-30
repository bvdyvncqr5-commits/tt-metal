# skillexp — machine a

generated 2026-07-30T09:27:26+00:00 on qb2-120-p05t03

tt-metal HEAD `5effae359ac` on
`mvasiljevic/qb2/skillexp/run/nofuse-advise`

## Phase 1 — functional decoder (owned by this machine)

| model | goal | gate | fd-ready tag |
|---|---|---|---|
| `microsoft_phi_3_5_mini_instruct` | complete | - | yes |
| `qwen_qwen3_6_27b` | complete | - | yes |
| `coherelabs_north_mini_code_1_0` | not-started | - | yes |
| `google_gemma_4_26b_a4b_it` | not-started | - | yes |

## Phase 2/3 — optimize, this machine's arms

| arm | model | goal | gate | done tag |
|---|---|---|---|---|
| nofuse-advise | `microsoft_phi_3_5_mini_instruct` | complete | pass | no |
| nofuse-advise | `qwen_qwen3_6_27b` | complete | pass | yes |
| nofuse-advise | `coherelabs_north_mini_code_1_0` | complete | pass | yes |
| nofuse-advise | `google_gemma_4_26b_a4b_it` | complete | pass | yes |
| fuse-advise | `microsoft_phi_3_5_mini_instruct` | complete | pass | yes |
| fuse-advise | `qwen_qwen3_6_27b` | complete | pass | yes |
| fuse-advise | `coherelabs_north_mini_code_1_0` | complete | pass | yes |
| fuse-advise | `google_gemma_4_26b_a4b_it` | complete | pass | yes |

## Device
```
tt-smi suppressed: measured stage may be live
```

## Cells taken over from the other machine (claimed)

| cell | claimed by | at | FD | done tag |
|---|---|---|---|---|
| — | none claimed | — | — | — |

A claim is one ref per cell; claiming is a **non-force** push of a new ref, so the remote decides
the winner and two machines cannot both proceed. The claim is released only once the cell is
tagged — a cell that ends untagged keeps its claim, so the owning machine does not blindly re-run
a failure someone else already hit.

## Rejected / contaminated cells and re-runs pending

| cell | why | parked at | state |
|---|---|---|---|
| `fuse-advise-microsoft_phi_3_5_mini_instruct` | inherited the sibling arm's artifacts (own advisor capture not run) | `mvasiljevic/qb2/skillexp/parked/CONTAMINATED-fuse-advise-microsoft_phi_3_5_mini_instruct` | tagged (check!) |
| `SUSPECT-nofuse-advise-microsoft_phi_3_5_mini_instruct` | parked | `mvasiljevic/qb2/skillexp/parked/SUSPECT-nofuse-advise-microsoft_phi_3_5_mini_instruct` | tag retracted -- **re-run queued** |
| `run-fuse-advise-with-contaminated-phi` | parked | `mvasiljevic/qb2/skillexp/parked/run-fuse-advise-with-contaminated-phi` | tag retracted -- **re-run queued** |

Re-runs are queued behind the live cell; a rejected cell is parked, never deleted, and gets no
tag until it passes `validate_cell` (own advisor capture, FD ancestor, no byte-identical
artifacts shared with the sibling arm).

## Blocked / critical (audit these before calling them real blockers)
- none
