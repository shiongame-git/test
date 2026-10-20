"""Reference 3D packing implementation with BL/PP decoders and GA search."""

from .bl import BLPriority, find_bl_point, pack
from .center_of_gravity import (
    CenterOfGravity,
    calculate_balance_distance,
    calculate_center_distance,
    calculate_center_of_gravity,
)
from ._decoder_config import DecoderConfig
from .errors import InputFormatError
from .evaluation import DecoderInput
from .execution import EvaluationStats, GAExecutionConfig
from .experiment import (
    BoxOrderMode,
    ComparisonDecoder,
    ComparisonProblemSignature,
    ComparisonTrialResult,
    ExperimentEnvironment,
    current_environment,
    order_boxes_for_experiment,
    run_comparison_trial,
)
from .ga import (
    EvaluationError,
    EvaluationResult,
    GAConfig,
    GAProgress,
    GARunResult,
    GenerationResult,
    Individual,
    evaluate_bl_individual,
    run_bl_ga,
)
from .io_box_sizes import read_box_sizes, parse_box_sizes_bytes
from .io_result_json import read_result_json, write_result_json
from .io_profile import read_decoder_profile_json, write_decoder_profile_json
from .model import Box, Container, PackingResult, Placement, Point
from .pp import pack_pp
from .pp_ga import (
    PPGARunResult,
    PPGenerationResult,
    evaluate_pp_individual,
    pp_state_from_result,
    run_pp_ga,
)
from .pp_state import PPState
from .profile import DecoderProfileMetrics, DecoderProfileRecord
from .unity_export import write_unity_result_data
from .validate import validate_result, validate_support

__all__ = [
    "BLPriority",
    "BoxOrderMode",
    "ComparisonDecoder",
    "ComparisonProblemSignature",
    "ComparisonTrialResult",
    "CenterOfGravity",
    "ExperimentEnvironment",
    "Box",
    "Container",
    "DecoderConfig",
    "DecoderProfileMetrics",
    "DecoderProfileRecord",
    "DecoderInput",
    "EvaluationError",
    "EvaluationResult",
    "EvaluationStats",
    "GAConfig",
    "GAExecutionConfig",
    "GAProgress",
    "GARunResult",
    "GenerationResult",
    "Individual",
    "InputFormatError",
    "PackingResult",
    "Placement",
    "Point",
    "PPGARunResult",
    "PPGenerationResult",
    "PPState",
    "calculate_center_distance",
    "calculate_balance_distance",
    "calculate_center_of_gravity",
    "current_environment",
    "evaluate_bl_individual",
    "evaluate_pp_individual",
    "find_bl_point",
    "pack",
    "pack_pp",
    "order_boxes_for_experiment",
    "pp_state_from_result",
    "read_box_sizes",
    "parse_box_sizes_bytes",
    "read_result_json",
    "read_decoder_profile_json",
    "run_bl_ga",
    "run_comparison_trial",
    "run_pp_ga",
    "validate_result",
    "validate_support",
    "write_result_json",
    "write_decoder_profile_json",
    "write_unity_result_data",
]

from .workers import WorkersRequest, detect_cpu, resolve_workers, preflight_workers
from .io_run_metadata import (RunMetadata, read_run_metadata_json, write_run_metadata_json, validate_run_metadata,
                              read_comparison_report_json, write_comparison_report_json, validate_comparison_report,
                              read_execution_records)
from .run_context import RunOptions, ReplaySettings, prepare_manual_replay, run_manual_replay
__all__ += ["WorkersRequest", "detect_cpu", "resolve_workers", "preflight_workers", "RunMetadata",
            "read_run_metadata_json", "write_run_metadata_json", "validate_run_metadata",
            "read_comparison_report_json", "write_comparison_report_json", "validate_comparison_report",
            "read_execution_records", "RunOptions", "ReplaySettings", "prepare_manual_replay", "run_manual_replay"]

from .research import (
    ResearchExperiment,
    ResearchRecord,
    read_generations_jsonl,
    read_research_experiment_json,
    read_research_record_json,
    validate_research_experiment,
    validate_research_record,
    write_research_experiment_json,
    write_research_record_json,
)
from .research_records import (
    ResearchCapture,
    capture_ga_run,
    capture_single_run,
    make_provenance,
    read_research_record,
    save_research_record,
    snapshot_for_single,
    snapshot_from_ga,
    utc_now,
)
from .research_experiment import (
    ResearchExperimentPlan,
    prepare_research_experiment,
    regenerate_research_experiment_csv,
    read_research_experiment,
    run_research_experiment,
)

__all__ += [
    "ResearchExperiment", "ResearchRecord", "ResearchCapture", "capture_ga_run", "capture_single_run",
    "make_provenance", "read_generations_jsonl", "read_research_experiment_json", "read_research_record",
    "read_research_record_json", "save_research_record", "snapshot_for_single", "snapshot_from_ga", "utc_now",
    "validate_research_experiment", "validate_research_record", "write_research_experiment_json",
    "write_research_record_json",
    "ResearchExperimentPlan", "prepare_research_experiment", "read_research_experiment",
    "regenerate_research_experiment_csv", "run_research_experiment",
]
