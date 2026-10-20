"""Headless glue shared by the Tk GUI and GUI/core regression tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from collections.abc import Callable, Iterable

from packing import (
    BLPriority,
    DecoderConfig,
    Box,
    Container,
    GAConfig,
    GAExecutionConfig,
    RunOptions,
    WorkersRequest,
    RunMetadata,
    GAProgress,
    GARunResult,
    PPGARunResult,
    PackingResult,
    run_bl_ga,
    run_pp_ga,
    write_result_json,
    write_unity_result_data,
)


def parse_decoder_settings(candidate_policy: str, support_percent: str) -> DecoderConfig:
    """Parse the GUI's visible decoder fields without silently changing them."""
    if type(support_percent) is not str:
        raise TypeError("support percentage must be entered as text")
    try:
        parsed_support_percent = int(support_percent)
    except ValueError as exc:
        raise ValueError("support percentage must be an integer from 0 to 100") from exc
    try:
        return DecoderConfig(
            candidate_policy=candidate_policy,
            candidate_policy_version=1,
            min_support_percent=parsed_support_percent,
            support_model_version="support-union-v1",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid decoder settings: {exc}") from exc


def run_ga_core(
    container: Container,
    boxes: Iterable[Box],
    *,
    decoder: str,
    priority: BLPriority,
    config: GAConfig,
    seed: int,
    execution: GAExecutionConfig,
    decoder_config: DecoderConfig | None = None,
    run_options: RunOptions | None = None,
    progress_callback: Callable[[GAProgress], None] | None = None,
) -> GARunResult | PPGARunResult:
    """Run the exact core call used by the GUI GA callback."""
    if decoder == "BL":
        runner = run_bl_ga
    elif decoder == "PP":
        runner = run_pp_ga
    else:
        raise ValueError("decoder must be BL or PP")
    return runner(
        container,
        boxes,
        priority=priority,
        config=config,
        seed=seed,
        execution=execution,
        decoder_config=decoder_config,
        progress_callback=progress_callback,
        run_options=run_options or RunOptions(origin="gui"),
    )


def render_previews(result: PackingResult) -> tuple[str, str]:
    """Render GUI previews through the same public JSON/Unity writers."""
    with tempfile.TemporaryDirectory(prefix="packing3d_gui_") as tmp:
        tmpdir = Path(tmp)
        json_path = tmpdir / "result.json"
        unity_path = tmpdir / "ResultData.txt"
        write_result_json(json_path, result)
        write_unity_result_data(unity_path, result)
        return (
            json_path.read_text(encoding="utf-8"),
            unity_path.read_text(encoding="utf-8"),
        )


def parse_workers_request(mode: str, text: str) -> WorkersRequest:
    """Auto/Max ignore the unused Manual widget, including invalid text."""
    mode_map = {"Auto": "auto", "Max": "max", "Manual": "manual"}
    if mode not in mode_map:
        raise ValueError("workers mode must be Auto, Max or Manual")
    value = int(text) if mode == "Manual" else None
    return WorkersRequest(mode_map[mode], value)


class RunRecordState:
    """Headless GUI save state; only a successful writer records a path."""
    def __init__(self):
        self.metadata = None
        self.saved_path = None

    def clear(self):
        self.metadata = None
        self.saved_path = None

    def complete(self, run):
        if type(run.run_metadata) is not RunMetadata:
            raise TypeError("completed GA needs run metadata")
        self.metadata = run.run_metadata
        self.saved_path = None

    def save(self, path, writer=None):
        from packing import write_run_metadata_json
        if self.metadata is None:
            raise ValueError("no completed GA record")
        if not path:
            return False
        (writer or write_run_metadata_json)(path, self.metadata)
        self.saved_path = Path(path)
        return True
