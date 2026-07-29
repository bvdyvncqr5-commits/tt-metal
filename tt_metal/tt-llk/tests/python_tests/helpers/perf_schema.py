# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


class PerfSchemaError(AssertionError):
    """
    Raised when a perf report accumulates more than one column schema in a
    single CSV (ragged, NaN-filled rows that break strict-JSON dashboards and
    the compare feature). Fail-loud so the contaminated CSV never ships.
    """


MARKER = "marker"

FORMAT_HEADERS = (
    "formats.input_A",
    "formats.input_B",
    "formats.register_A",
    "formats.register_B",
    "formats.output",
)
FLAG_HEADERS = ("unpack_to_dest", "dest_acc")

LOOP_FACTOR_COLUMN = "loop_factor"
TILE_CNT_COLUMN = "tile_cnt"
TEST_NAME_COLUMN = "test_name"

MEAN = "mean"
STD = "std"
TEXT_SIZE_PREFIX = "TEXT_SIZE("


def stat_prefix(kind: str) -> str:
    """Prefix of a stat column, e.g. stat_prefix("mean") -> "mean(" (for startswith)."""
    return f"{kind}("


def stat_column(base: str, kind: str) -> str:
    """Stat column, e.g. stat_column("L1_TO_L1", "mean") -> "mean(L1_TO_L1)"."""
    return f"{stat_prefix(kind)}{base})"


def metric_column(run_type_name: str, base: str) -> str:
    """Run-type-prefixed column, e.g. metric_column("L1_TO_L1", "x") -> "L1_TO_L1_x"."""
    return f"{run_type_name}_{base}"


def text_size_column(run_type_name: str) -> str:
    """Kernel code-size column, e.g. text_size_column("L1_TO_L1") -> "TEXT_SIZE(L1_TO_L1)"."""
    return f"{TEXT_SIZE_PREFIX}{run_type_name})"


def counter_base(bank: str, counter: str) -> str:
    """Raw HW-counter base name, e.g. counter_base("FPU", "FPU_COUNTER") -> "FPU.FPU_COUNTER"."""
    return f"{bank}.{counter}"


def cycles_of(base: str) -> str:
    """Cycles variant of a counter base, e.g. cycles_of("FPU.FPU_COUNTER") -> "FPU.FPU_COUNTER.cycles"."""
    return f"{base}.cycles"


# Golden CSV-header catalog
#
# This catalog is HAND-MAINTAINED on purpose: a header changes ONLY when someone
# edits a set below, so a rename becomes a reviewed diff instead of a silent
# drift. When a test legitimately adds or renames a header, update the matching
# set below in the SAME pull request.

# Run-type names — mirror helpers/llk_params.py::PerfRunType.
RUN_TYPE_NAMES = frozenset(
    {"L1_TO_L1", "UNPACK_ISOLATE", "MATH_ISOLATE", "PACK_ISOLATE", "L1_CONGESTION"}
)

# Stat kinds a timing/counter column may carry.
STAT_KINDS = (MEAN, STD)

# Non-sweep key columns present in a report.
KEY_COLUMNS = (MARKER, TEST_NAME_COLUMN, LOOP_FACTOR_COLUMN, TILE_CNT_COLUMN)

# Derived efficiency metric base names — mirror the ``*_pct`` keys that
# helpers/metrics.py::compute_metrics exports (the only keys _exportable() keeps).
METRIC_BASES = frozenset(
    {
        "fpu_utilization_pct",
        "compute_utilization_pct",
        "unpack_thread_stall_pct",
        "math_thread_stall_pct",
        "pack_thread_stall_pct",
        "math_sem_wait_pct",
        "pack_sem_wait_pct",
        "unpack0_write_eff_pct",
        "unpack1_write_eff_pct",
        "unpack_write_eff_pct",
        "unpack_to_math_flow0_pct",
        "unpack_to_math_flow1_pct",
        "unpack_to_math_flow_pct",
        "pack_utilization_pct",
        "pack_dest_eff_pct",
        "fidelity_stall_pct",
        "math_src_stall_pct",
    }
)

# Sweep-parameter field names — one per TemplateParameter/RuntimeParameter field
# that becomes a CSV header. Every name must be unique across all parameter
# classes (a duplicate would collide two columns into one header).
GOLDEN_SWEEP_PARAMS = frozenset(
    {
        "acc_to_dest",
        "add_top_row",
        "alpha_bits",
        "approx_mode",
        "bcast_dim",
        "beta_bits",
        "block_ct_dim",
        "block_rt_dim",
        "broadcast_type",
        "c_dimm",
        "clamp_negative",
        "clobber_op",
        "configure_test_run_idx",
        "count",
        "ct_dim",
        "data_copy_type",
        "data_format",
        "dest_sync",
        "disable_src_zero_flag",
        "do_restore",
        "dst_index",
        "dst_tile_idx",
        "enable_2x_format",
        "enable_direct_indexing",
        "face_c_dim",
        "face_r_dim",
        "fast_mode",
        "filler_count",
        "full_ct_dim",
        "full_rt_dim",
        "host_is_stream_consumer",
        "host_is_stream_producer",
        "implied_math_format",
        "in0_c_dim",
        "in0_face_c_dim",
        "in0_face_r_dim",
        "in0_r_dim",
        "in1_c_dim",
        "in1_face_c_dim",
        "in1_face_r_dim",
        "in1_r_dim",
        "input_format",
        "input_num_blocks",
        "input_num_tiles_in_block",
        "input_tile_cnt",
        "is_max_op",
        "is_reduce_to_one",
        "iterations",
        "k_dimm",
        "l1_acc",
        "loop_factor",
        "math_fidelity",
        "math_transpose_faces",
        "mathop",
        "narrow_tile",
        "nest_depth",
        "num_blocks",
        "num_faces",
        "num_faces_A",
        "num_faces_B",
        "num_faces_c_dim_A",
        "num_faces_c_dim_B",
        "num_faces_r_dim_A",
        "num_faces_r_dim_B",
        "num_rows_to_pack",
        "num_tiles_in_block",
        "offset",
        "op",
        "output_format",
        "output_num_blocks",
        "output_num_tiles_in_block",
        "output_tile_cnt",
        "overlap_first_half",
        "partial_a",
        "partial_b",
        "partial_face_math",
        "partial_face_pack",
        "perf_run_type",
        "polluter",
        "pool_type",
        "r_dimm",
        "reduce_block_ct_dim",
        "reduce_pool_type",
        "reinit_mode",
        "relu_config",
        "respect_trigger",
        "reuse_dest_type",
        "sequence",
        "src0_tile_idx",
        "src1_tile_idx",
        "srca_reuse_count",
        "stable_sort",
        "stochastic_rounding",
        "throttle_level",
        "tile_cnt",
        "tile_index",
        "tilize",
        "to_from_int8",
        "topk_k",
        "topk_matrix_width",
        "topk_sort_direction",
        "topk_stable_sort",
        "unary_extra",
        "unpack_transpose_faces",
        "unpack_transpose_within_face",
        "unpacker_engine_sel",
        "use_runtime",
        "value_bits",
        "vector_mode",
        "via_reconfig",
        "victim_face_r_dim",
        "victim_num_faces",
    }
)
