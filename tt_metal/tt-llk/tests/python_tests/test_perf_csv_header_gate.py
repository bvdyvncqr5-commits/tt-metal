# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Merge gate: perf-CSV header names must not drift from the golden catalog.

The hand-maintained catalog in ``helpers/perf_schema.py`` is the single source
of truth for perf-CSV header names. These tests read the LIVE source with
``ast`` and fail if it drifts from the catalog, so a header rename cannot merge
without a deliberate catalog edit. They parse the source only, so they need no
hardware and run in any CI lane.
"""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAM_BASES = {"TemplateParameter", "RuntimeParameter"}


def _iter_source_files():
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        yield path


def _load_perf_schema():
    """Load perf_schema.py directly (it has no imports), bypassing the helpers
    package __init__ which pulls device libraries. Keeps this runnable anywhere."""
    path = ROOT / "helpers" / "perf_schema.py"
    spec = importlib.util.spec_from_file_location("_perf_schema_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_parameter_fields() -> dict:
    field_owners: dict[str, list] = {}
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(
                isinstance(base, ast.Name) and base.id in PARAM_BASES
                for base in node.bases
            ):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    field = stmt.target.id
                    if field.startswith("_"):
                        continue
                    field_owners.setdefault(field, []).append(
                        (node.name, path.relative_to(ROOT).as_posix())
                    )
    return field_owners


def _enum_member_names(module_filename: str, enum_name: str) -> set:
    """Names assigned in an Enum class body, read statically (no import)."""
    path = ROOT / "helpers" / module_filename
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == enum_name:
            return {
                target.id
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                for target in stmt.targets
                if isinstance(target, ast.Name)
            }
    return set()


def test_sweep_params_match_golden_catalog():
    ps = _load_perf_schema()
    live = set(_collect_parameter_fields())
    golden = set(ps.GOLDEN_SWEEP_PARAMS)

    added = sorted(live - golden)
    removed = sorted(golden - live)

    lines = [
        "Perf-CSV sweep headers drifted from the golden catalog "
        "(perf_schema.GOLDEN_SWEEP_PARAMS).",
        "",
        f"  New/renamed field(s) in source, NOT in catalog: {added}",
        f"  Field(s) in catalog, gone from source:          {removed}",
        "",
        "A header is a join key between the branch report and the master "
        "baseline, so a silent rename breaks the compare. If this change is "
        "intentional, update GOLDEN_SWEEP_PARAMS in the same PR.",
    ]
    assert not added and not removed, "\n".join(lines)


def test_run_type_names_match_source():
    ps = _load_perf_schema()
    live = _enum_member_names("llk_params.py", "PerfRunType")
    assert live == set(ps.RUN_TYPE_NAMES), (
        f"PerfRunType members {sorted(live)} drifted from "
        f"perf_schema.RUN_TYPE_NAMES {sorted(ps.RUN_TYPE_NAMES)}. Update the "
        f"catalog: a run-type name prefixes every metric/counter header."
    )


def test_metric_bases_match_source():
    """The catalog's metric bases must equal the *_pct dict keys metrics.py exports."""
    tree = ast.parse((ROOT / "helpers" / "metrics.py").read_text())
    # export_metrics keeps exactly the metric-dict keys ending in "_pct" (see
    # _exportable()). Read the dict keys via ast, not a text scan, so an
    # unrelated "_pct" string literal (log line, docstring, m.get() arg) can
    # neither trip nor evade the gate.
    live = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and key.value.endswith("_pct")
    }
    ps = _load_perf_schema()
    assert live == set(ps.METRIC_BASES), (
        f"Efficiency metric names drifted. In source but not catalog: "
        f"{sorted(live - set(ps.METRIC_BASES))}; in catalog but not source: "
        f"{sorted(set(ps.METRIC_BASES) - live)}. Update perf_schema.METRIC_BASES."
    )
