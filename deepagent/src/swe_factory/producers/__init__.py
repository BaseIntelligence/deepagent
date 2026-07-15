"""Candidate producers: synthetic_grounded, real_pr, and Harbor DeepAgent motors."""

from swe_factory.producers.harbor_labeling import (
    DualRunLabels,
    HarborLabelError,
    SuiteOutcome,
    compute_dual_run_labels,
    label_cohorts_from_outcomes,
    labels_from_suite_outcomes,
    write_tests_config_json,
)
from swe_factory.producers.harbor_motors import (
    HARD_MULTI_FILE_FLOOR,
    MOTOR_SEEDS,
    HarborMaterials,
    HarborMotorError,
    HarborMotorResult,
    HarborMotorSeed,
    get_motor_seed,
    list_motor_seeds,
    produce_all_offline_motors,
    produce_harbor_materials,
    produce_harbor_pack,
)
from swe_factory.producers.hard_filter import (
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
    SOFT_MULTI_FILE_FLOOR,
    HardFilterResult,
    HardFilterStats,
    apply_product_hard_filter,
    evaluate_product_hard_filter,
    measure_source_hunk_count,
    reject_ledger_row,
)

# materialize_from_pr is importable as a submodule only — not re-exported here —
# to avoid a circular import:
#   sources.discover → producers (this package) → materialize_from_pr → sources.discover
# Import: ``from swe_factory.producers.materialize_from_pr import …``
from swe_factory.producers.pr_miner import (
    MergedPR,
    PrFileChange,
    PrMineError,
    PrMiner,
    RealPrCandidate,
    multi_file_source_filter,
)
from swe_factory.producers.pr_miner import (
    produce_offline_fixture as produce_real_pr_offline_fixture,
)
from swe_factory.producers.real_dual_run import (
    RealDualRunError,
    RealDualRunResult,
    label_real_pr_dual_run,
    label_real_pr_from_outcomes,
)
from swe_factory.producers.suite_reporters import (
    get_suite_reporter,
    list_reporter_languages,
    reporter_info,
)
from swe_factory.producers.synth import (
    MUTATION_FUNCTION_REMOVAL,
    MUTATION_MULTI_FAULT,
    SynthCandidate,
    SynthError,
    SynthProducer,
    build_problem_statement,
    produce_from_green_fixture,
)

__all__ = [
    "HARD_MULTI_FILE_FLOOR",
    "MOTOR_SEEDS",
    "MUTATION_FUNCTION_REMOVAL",
    "MUTATION_MULTI_FAULT",
    "PRODUCT_MULTI_FILE_FLOOR",
    "PRODUCT_SOURCE_HUNK_FLOOR",
    "SOFT_MULTI_FILE_FLOOR",
    "DualRunLabels",
    "HardFilterResult",
    "HardFilterStats",
    "HarborLabelError",
    "HarborMaterials",
    "HarborMotorError",
    "HarborMotorResult",
    "HarborMotorSeed",
    "MergedPR",
    "PrFileChange",
    "PrMineError",
    "PrMiner",
    "RealDualRunError",
    "RealDualRunResult",
    "RealPrCandidate",
    "SuiteOutcome",
    "SynthCandidate",
    "SynthError",
    "SynthProducer",
    "apply_product_hard_filter",
    "build_problem_statement",
    "compute_dual_run_labels",
    "evaluate_product_hard_filter",
    "get_motor_seed",
    "get_suite_reporter",
    "label_cohorts_from_outcomes",
    "label_real_pr_dual_run",
    "label_real_pr_from_outcomes",
    "labels_from_suite_outcomes",
    "list_motor_seeds",
    "list_reporter_languages",
    "measure_source_hunk_count",
    "multi_file_source_filter",
    "produce_all_offline_motors",
    "produce_from_green_fixture",
    "produce_harbor_materials",
    "produce_harbor_pack",
    "produce_real_pr_offline_fixture",
    "reject_ledger_row",
    "reporter_info",
    "write_tests_config_json",
]
