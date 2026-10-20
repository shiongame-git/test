"""Windows-oriented GUI for the 3D Bin Packing reference implementation.

This module is intentionally a thin front end. It delegates parsing, BL/GA
execution, validation, JSON I/O, and Unity export to the public ``packing`` API.
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import replace
import queue
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4

from gui_core import (
    RunRecordState,
    parse_decoder_settings,
    parse_workers_request,
    render_previews,
    run_ga_core,
)
from research_gui import ResearchExperimentDialog

from packing import (
    BLPriority,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    RunOptions,
    prepare_manual_replay,
    read_execution_records,
    detect_cpu,
    resolve_workers,
    GAProgress,
    GARunResult,
    PPGARunResult,
    PackingResult,
    capture_ga_run,
    capture_single_run,
    make_provenance,
    pack,
    parse_box_sizes_bytes,
    read_result_json,
    validate_result,
    validate_support,
    write_result_json,
    write_unity_result_data,
    save_research_record,
    snapshot_for_single,
    utc_now,
)
from packing.research_records import default_research_directory, validate_research_output_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "box_sizes_cube.txt"


class PackingDemoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("3D Bin Packing Python参照実装")
        self.root.geometry("1180x820")
        self.root.minsize(900, 620)

        self.container = None
        self.boxes = None
        self.result: PackingResult | None = None
        self.result_origin: str | None = None
        self.result_expected_boxes = None
        self.result_decoder_config: DecoderConfig | None = None
        self.ga_result: GARunResult | PPGARunResult | None = None
        self.source_path: Path | None = None
        self.fixture_sha256 = None
        self.source_bytes: bytes | None = None
        self.research_capture = None
        self.record_state = RunRecordState()
        self.replay_settings = None
        self.problem_options = RunOptions(origin="gui")
        self._applying_replay = False
        self._pack_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._running = False
        self._ga_started_at: float | None = None
        self._last_ga_progress: GAProgress | None = None
        self._research_experiment_dialog = None
        self._research_experiment_active = False
        self._close_when_experiment_done = False

        self.source_var = tk.StringVar(value="未選択")
        self.container_var = tk.StringVar(value="—")
        self.box_count_var = tk.StringVar(value="—")
        self.status_var = tk.StringVar(value="入力ファイルまたは結果JSONを選択してください。")
        self.placed_var = tk.StringVar(value="—")
        self.failed_var = tk.StringVar(value="—")
        self.volume_var = tk.StringVar(value="—")
        self.density_var = tk.StringVar(value="—")
        self.validation_var = tk.StringVar(value="未検証")
        self.priority_var = tk.StringVar(value=BLPriority.BOTTOM_FIRST.value)
        self.candidate_policy_var = tk.StringVar(value="Cartesian")
        self.min_support_percent_var = tk.StringVar(value="55")
        self.decoder_info_var = tk.StringVar(value="decoder: Cartesian v1 / 支持55%")
        self.ga_decoder_var = tk.StringVar(value="BL")
        default_ga = GAConfig()
        self.ga_population_var = tk.StringVar(value=str(default_ga.population_size))
        self.ga_generations_var = tk.StringVar(value=str(default_ga.generation_count))
        self.ga_crossover_var = tk.StringVar(value=str(default_ga.crossover_rate))
        self.ga_mutation_var = tk.StringVar(value=str(default_ga.mutation_rate))
        self.ga_cog_var = tk.BooleanVar(value=default_ga.use_center_of_gravity)
        self.ga_cog_lambda_var = tk.StringVar(value=str(default_ga.center_of_gravity_lambda))
        self.ga_seed_var = tk.StringVar(value="0")
        self.ga_workers_mode_var = tk.StringVar(value="Auto")
        self.workers_info_var = tk.StringVar(value="ワーカー数: 未確認")
        self.record_info_var = tk.StringVar(value="実行情報: —")
        self.ga_workers_var = tk.StringVar(value="1")
        self.ga_cache_var = tk.BooleanVar(value=False)
        self.ga_info_var = tk.StringVar(value="実行情報: —")
        self.ga_progress_percent_var = tk.DoubleVar(value=0.0)
        self.ga_progress_var = tk.StringVar(value="GA進捗: —")

        self._build_ui()
        if hasattr(self.root, "protocol"):
            self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self._set_result_actions(False)
        self.pack_button.state(["disabled"])
        self.ga_button.state(["disabled"])
        for variable in (self.ga_workers_mode_var, self.ga_workers_var, self.ga_population_var,
                         self.ga_generations_var, self.ga_crossover_var, self.ga_mutation_var,
                         self.ga_cog_var, self.ga_cog_lambda_var,
                         self.ga_seed_var, self.ga_decoder_var, self.priority_var, self.ga_cache_var,
                         self.candidate_policy_var, self.min_support_percent_var):
            variable.trace_add("write", self._on_settings_changed)
        self._on_settings_changed()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        title = ttk.Label(outer, text="3D Bin Packing Python参照実装", font=("Segoe UI", 18, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        input_frame = ttk.LabelFrame(outer, text="入力", padding=10)
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.columnconfigure(4, weight=1)

        self.open_input_button = ttk.Button(
            input_frame, text="box_sizes.txt を選択", command=self.open_box_sizes
        )
        self.open_input_button.grid(row=0, column=0, padx=(0, 8), pady=(0, 8), sticky="w")

        self.sample_button = ttk.Button(input_frame, text="サンプルを使用", command=self.use_sample)
        self.sample_button.grid(row=0, column=1, padx=(0, 8), pady=(0, 8), sticky="w")

        self.open_json_button = ttk.Button(input_frame, text="結果JSONを開く", command=self.open_result_json)
        self.open_json_button.grid(row=0, column=2, padx=(0, 8), pady=(0, 8), sticky="w")
        self.open_metadata_button = ttk.Button(input_frame, text="実行情報から設定を読み込む", command=self.open_run_metadata)
        self.open_metadata_button.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        priority_frame = ttk.Frame(input_frame)
        priority_frame.grid(row=0, column=3, padx=(8, 12), pady=(0, 8), sticky="e")
        ttk.Label(priority_frame, text="候補点優先順:").pack(side="left", padx=(0, 6))
        self.priority_combo = ttk.Combobox(
            priority_frame,
            textvariable=self.priority_var,
            values=tuple(priority.value for priority in BLPriority),
            state="readonly",
            width=13,
        )
        self.priority_combo.pack(side="left")

        self.pack_button = ttk.Button(input_frame, text="BL配置を実行", command=self.start_pack)
        self.pack_button.grid(row=0, column=4, pady=(0, 8), sticky="e")

        info = ttk.Frame(input_frame)
        info.grid(row=1, column=0, columnspan=5, sticky="ew")
        info.columnconfigure(1, weight=1)
        ttk.Label(info, text="入力元:").grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.source_var).grid(row=0, column=1, sticky="w", padx=(8, 20))
        ttk.Label(info, text="コンテナ:").grid(row=0, column=2, sticky="w")
        ttk.Label(info, textvariable=self.container_var).grid(row=0, column=3, sticky="w", padx=(8, 20))
        ttk.Label(info, text="箱数:").grid(row=0, column=4, sticky="w")
        ttk.Label(info, textvariable=self.box_count_var).grid(row=0, column=5, sticky="w", padx=(8, 0))

        ga_frame = ttk.LabelFrame(input_frame, text="GA（主要フロー）", padding=8)
        ga_frame.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(8, 0))
        decoder_field = ttk.Frame(ga_frame)
        decoder_field.grid(row=0, column=0, padx=(0, 12), sticky="w")
        ttk.Label(decoder_field, text="decoder:").pack(side="left", padx=(0, 4))
        self.ga_decoder_combo = ttk.Combobox(
            decoder_field,
            textvariable=self.ga_decoder_var,
            values=("BL", "PP"),
            state="readonly",
            width=5,
        )
        self.ga_decoder_combo.pack(side="left")

        ga_fields = (
            ("個体数", self.ga_population_var, 6),
            ("世代数", self.ga_generations_var, 6),
            ("交叉率", self.ga_crossover_var, 7),
            ("突然変異率", self.ga_mutation_var, 7),
            ("seed", self.ga_seed_var, 8),
        )
        self.ga_entries = []
        for offset, (label, variable, width) in enumerate(ga_fields, start=1):
            field = ttk.Frame(ga_frame)
            field.grid(row=0, column=offset, padx=(0, 12), sticky="w")
            ttk.Label(field, text=f"{label}:").pack(side="left", padx=(0, 4))
            entry = ttk.Entry(field, textvariable=variable, width=width)
            entry.pack(side="left")
            self.ga_entries.append(entry)
        button_column = len(ga_fields) + 1
        ga_frame.columnconfigure(button_column, weight=1)
        self.ga_button = ttk.Button(ga_frame, text="GAを実行", command=self.start_ga)
        self.ga_button.grid(row=0, column=button_column, sticky="e")
        self.experiment_button = ttk.Button(
            ga_frame, text="Research Experiment…", command=self.open_research_experiment
        )
        self.experiment_button.grid(row=6, column=0, columnspan=button_column + 1, sticky="w", pady=(8, 0))
        self.experiment_button.state(["disabled"])

        decoder_settings_frame = ttk.Frame(ga_frame)
        decoder_settings_frame.grid(row=1, column=0, columnspan=button_column + 1, sticky="w", pady=(8, 0))
        ttk.Label(decoder_settings_frame, text="candidate policy:").pack(side="left", padx=(0, 4))
        self.candidate_policy_combo = ttk.Combobox(
            decoder_settings_frame,
            textvariable=self.candidate_policy_var,
            values=("Cartesian", "LocalNeighbors"),
            state="readonly",
            width=16,
        )
        self.candidate_policy_combo.pack(side="left", padx=(0, 12))
        ttk.Label(decoder_settings_frame, text="最小支持率:").pack(side="left", padx=(0, 4))
        self.min_support_percent_entry = ttk.Entry(
            decoder_settings_frame, textvariable=self.min_support_percent_var, width=5
        )
        self.min_support_percent_entry.pack(side="left", padx=(0, 4))
        ttk.Label(decoder_settings_frame, text="%").pack(side="left", padx=(0, 12))
        ttk.Label(decoder_settings_frame, textvariable=self.decoder_info_var).pack(side="left")

        execution_frame = ttk.Frame(ga_frame)
        execution_frame.grid(row=2, column=0, columnspan=button_column + 1, sticky="w", pady=(8, 0))
        ttk.Label(execution_frame, text="評価ワーカー数:").pack(side="left", padx=(0, 4))
        self.ga_workers_mode_combo = ttk.Combobox(execution_frame, textvariable=self.ga_workers_mode_var,
                                                 values=("Auto", "Max", "Manual"), state="readonly", width=8)
        self.ga_workers_mode_combo.pack(side="left", padx=(0, 8))
        self.ga_workers_entry = ttk.Entry(
            execution_frame, textvariable=self.ga_workers_var, width=6
        )
        self.ga_workers_entry.pack(side="left", padx=(0, 12))
        self.ga_cache_check = ttk.Checkbutton(
            execution_frame, text="評価結果をキャッシュする", variable=self.ga_cache_var
        )
        self.ga_cache_check.pack(side="left", padx=(0, 14))
        self.ga_cog_check = ttk.Checkbutton(
            execution_frame, text="重心を評価に使う", variable=self.ga_cog_var
        )
        self.ga_cog_check.pack(side="left", padx=(0, 8))
        ttk.Label(execution_frame, text="λ:").pack(side="left", padx=(0, 4))
        self.ga_cog_lambda_entry = ttk.Entry(
            execution_frame, textvariable=self.ga_cog_lambda_var, width=7
        )
        self.ga_cog_lambda_entry.pack(side="left", padx=(0, 14))
        ttk.Label(execution_frame, text="重量: Unity方式 (ID別1〜10 / seed固定)").pack(
            side="left", padx=(0, 12)
        )
        self.ga_paper_defaults_button = ttk.Button(
            execution_frame, text="GA標準設定", command=self.apply_paper_ga_defaults
        )
        self.ga_paper_defaults_button.pack(side="left", padx=(0, 8))
        self.ga_quick_button = ttk.Button(
            execution_frame, text="Quick test", command=self.apply_quick_ga_defaults
        )
        self.ga_quick_button.pack(side="left")
        progress_frame = ttk.Frame(ga_frame)
        progress_frame.grid(row=3, column=0, columnspan=button_column + 1, sticky="ew", pady=(8, 0))
        progress_frame.columnconfigure(0, weight=1)
        self.ga_progress_bar = ttk.Progressbar(
            progress_frame,
            maximum=100.0,
            variable=self.ga_progress_percent_var,
            mode="determinate",
        )
        self.ga_progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        ttk.Label(progress_frame, textvariable=self.ga_progress_var).grid(row=0, column=1, sticky="e")
        ttk.Label(ga_frame, textvariable=self.workers_info_var, wraplength=1080).grid(
            row=4, column=0, columnspan=button_column + 1, sticky="w", pady=(5, 0))
        ttk.Label(ga_frame, text="AutoはCPU推定値・個体数・上限から決定。小規模問題ではManual 1が速い場合があります。").grid(
            row=5, column=0, columnspan=button_column + 1, sticky="w")

        summary = ttk.LabelFrame(outer, text="配置結果", padding=10)
        summary.grid(row=2, column=0, sticky="ew", pady=(10, 10))
        for col in range(8):
            summary.columnconfigure(col, weight=1 if col in (1, 3, 5, 7) else 0)

        ttk.Label(summary, text="配置成功").grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.placed_var, font=("Segoe UI", 12, "bold")).grid(
            row=0, column=1, sticky="w", padx=(8, 18)
        )
        ttk.Label(summary, text="配置失敗").grid(row=0, column=2, sticky="w")
        ttk.Label(summary, textvariable=self.failed_var, font=("Segoe UI", 12, "bold")).grid(
            row=0, column=3, sticky="w", padx=(8, 18)
        )
        ttk.Label(summary, text="配置体積").grid(row=0, column=4, sticky="w")
        ttk.Label(summary, textvariable=self.volume_var, font=("Segoe UI", 12, "bold")).grid(
            row=0, column=5, sticky="w", padx=(8, 18)
        )
        ttk.Label(summary, text="密度").grid(row=0, column=6, sticky="w")
        ttk.Label(summary, textvariable=self.density_var, font=("Segoe UI", 12, "bold")).grid(
            row=0, column=7, sticky="w", padx=(8, 0)
        )

        ttk.Label(summary, textvariable=self.ga_info_var).grid(
            row=1, column=0, columnspan=8, sticky="w", pady=(8, 0)
        )

        actions = ttk.Frame(summary)
        actions.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        self.validate_button = ttk.Button(actions, text="結果を検証", command=self.validate_current_result)
        self.validate_button.pack(side="left")
        ttk.Label(actions, textvariable=self.validation_var).pack(side="left", padx=(10, 24))
        self.save_json_button = ttk.Button(actions, text="結果JSONを保存", command=self.save_result_json)
        self.save_json_button.pack(side="left", padx=(0, 8))
        self.save_unity_button = ttk.Button(actions, text="Unity用データを保存", command=self.save_unity_data)
        self.save_unity_button.pack(side="left", padx=(0, 8))
        self.copy_unity_button = ttk.Button(
            actions, text="Unity形式をコピー", command=self.copy_unity_data
        )
        self.copy_unity_button.pack(side="left")
        self.save_metadata_button = ttk.Button(actions, text="実行情報を保存", command=self.save_run_metadata)
        self.save_metadata_button.pack(side="left", padx=(8, 0))
        self.save_metadata_button.state(["disabled"])
        self.research_save_button = ttk.Button(
            actions, text="研究記録として保存", command=self.save_research_record
        )
        self.research_save_button.pack(side="left", padx=(8, 0))
        self.research_save_button.state(["disabled"])
        ttk.Label(summary, textvariable=self.record_info_var, wraplength=1080).grid(
            row=3, column=0, columnspan=8, sticky="w", pady=(5, 0))

        notebook = ttk.Notebook(outer)
        notebook.grid(row=3, column=0, sticky="nsew")

        placements_tab = ttk.Frame(notebook, padding=6)
        preview_json_tab = ttk.Frame(notebook, padding=6)
        preview_unity_tab = ttk.Frame(notebook, padding=6)
        notebook.add(placements_tab, text="配置一覧")
        notebook.add(preview_json_tab, text="JSONプレビュー")
        notebook.add(preview_unity_tab, text="Unityプレビュー")

        placements_tab.rowconfigure(0, weight=1)
        placements_tab.columnconfigure(0, weight=1)
        columns = ("id", "size", "position", "state")
        self.tree = ttk.Treeview(placements_tab, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("id", text="ID")
        self.tree.heading("size", text="サイズ")
        self.tree.heading("position", text="座標")
        self.tree.heading("state", text="状態")
        self.tree.column("id", width=90, anchor="center", stretch=False)
        self.tree.column("size", width=180, anchor="center")
        self.tree.column("position", width=220, anchor="center")
        self.tree.column("state", width=120, anchor="center", stretch=False)
        tree_scroll = ttk.Scrollbar(placements_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")

        self.json_preview = self._make_text_preview(preview_json_tab)
        self.unity_preview = self._make_text_preview(preview_unity_tab)

        status = ttk.Label(outer, textvariable=self.status_var, anchor="w")
        status.grid(row=4, column=0, sticky="ew", pady=(8, 0))

    @staticmethod
    def _make_text_preview(parent: ttk.Frame) -> tk.Text:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        text = tk.Text(parent, wrap="none", font=("Consolas", 10), state="disabled")
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        return text

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _set_input_actions(self, enabled: bool) -> None:
        enabled = enabled and not self._research_experiment_active
        buttons = (self.open_input_button, self.sample_button, self.open_json_button, self.open_metadata_button)
        for button in buttons:
            button.state(["!disabled"] if enabled else ["disabled"])
        self.priority_combo.configure(state="readonly" if enabled else "disabled")
        self.ga_decoder_combo.configure(state="readonly" if enabled else "disabled")
        self.candidate_policy_combo.configure(state="readonly" if enabled else "disabled")
        self.min_support_percent_entry.configure(state="normal" if enabled else "disabled")
        for entry in self.ga_entries:
            entry.configure(state="normal" if enabled else "disabled")
        self.ga_workers_mode_combo.configure(state="readonly" if enabled else "disabled")
        self.ga_workers_entry.configure(state="normal" if enabled and self.ga_workers_mode_var.get() == "Manual" else "disabled")
        self.ga_cache_check.configure(state="normal" if enabled else "disabled")
        self.ga_paper_defaults_button.state(["!disabled"] if enabled else ["disabled"])
        self.ga_quick_button.state(["!disabled"] if enabled else ["disabled"])
        self.experiment_button.state(["!disabled"] if enabled and self.container is not None and self.boxes is not None else ["disabled"])

    def open_research_experiment(self) -> None:
        if self._running or self._research_experiment_dialog is not None or self.container is None or self.boxes is None:
            return
        self._research_experiment_dialog = ResearchExperimentDialog(self)

    def close_window(self) -> None:
        dialog = self._research_experiment_dialog
        if dialog is not None and dialog.worker is not None:
            self._close_when_experiment_done = True
            dialog.request_close()
            return
        if dialog is not None:
            dialog.destroy()
        self.root.destroy()

    def apply_paper_ga_defaults(self) -> None:
        config = GAConfig()
        self.ga_population_var.set(str(config.population_size))
        self.ga_generations_var.set(str(config.generation_count))
        self.ga_crossover_var.set(str(config.crossover_rate))
        self.ga_mutation_var.set(str(config.mutation_rate))
        self.ga_cog_var.set(config.use_center_of_gravity)
        self.ga_cog_lambda_var.set(str(config.center_of_gravity_lambda))
        self.ga_seed_var.set("0")
        self.status_var.set("GA設定を標準値に戻しました（seed=0）。")

    def apply_quick_ga_defaults(self) -> None:
        self.ga_population_var.set("4")
        self.ga_generations_var.set("2")
        self.ga_crossover_var.set("0.8")
        self.ga_mutation_var.set("0.1")
        self.ga_cog_var.set(False)
        self.ga_cog_lambda_var.set(str(GAConfig().center_of_gravity_lambda))
        self.ga_seed_var.set("0")
        self.status_var.set("GA設定を簡易テスト用の軽量値へ変更しました。")

    @staticmethod
    def _workers_description(execution, environment, prefix):
        w = execution["workers"]
        cpu = environment["cpu"]["detection"]
        reasons = w["limit_reasons"] + w["warning_reasons"] + cpu["fallback_reasons"]
        return (f"workers {prefix}: {w['requested_mode']} → {w['resolved_workers']} / "
                f"CPU推定={cpu['available_logical_estimate']} ({cpu['selected_method']}) / "
                f"理由={', '.join(reasons) or 'なし'}")

    def _on_settings_changed(self, *_):
        if self._applying_replay or self._running:
            return
        if self.replay_settings is not None:
            self.replay_settings = None
            self.status_var.set("設定を変更したため、保存した条件での再実行ではなくなりました。")
        self.ga_workers_entry.configure(state="normal" if self.ga_workers_mode_var.get() == "Manual" else "disabled")
        self.ga_cog_lambda_entry.configure(state="normal" if self.ga_cog_var.get() else "disabled")
        try:
            decoder_config = parse_decoder_settings(
                self.candidate_policy_var.get(), self.min_support_percent_var.get()
            )
            self.decoder_info_var.set(
                f"decoder: {decoder_config.candidate_policy} v{decoder_config.candidate_policy_version} / "
                f"支持{decoder_config.min_support_percent}% ({decoder_config.support_model_version})"
            )
        except (TypeError, ValueError) as exc:
            self.decoder_info_var.set(f"配置設定を確認できません: {exc}")
        try:
            from packing.workers import executor_limit
            request = parse_workers_request(self.ga_workers_mode_var.get(), self.ga_workers_var.get())
            cpu = detect_cpu()
            w = resolve_workers(request, cpu["available_logical_estimate"], int(self.ga_population_var.get()), executor_limit())
            self.workers_info_var.set(self._workers_description({"workers": w}, {"cpu": {"detection": cpu}}, "実行前の推定値"))
        except (TypeError, ValueError) as exc:
            self.workers_info_var.set(f"ワーカー設定を確認できません: {exc}")

    def save_run_metadata(self):
        metadata = self.record_state.metadata
        if self._running or metadata is None:
            return
        filename = filedialog.asksaveasfilename(
            title="実行情報を保存", initialdir=str(self.source_path.parent if self.source_path else PROJECT_ROOT),
            initialfile=f"ga-{metadata['run_id']}.run.json", defaultextension=".run.json",
            filetypes=(("JSONファイル", "*.json"), ("すべてのファイル", "*.*")))
        if not filename:
            return
        try:
            self.record_state.save(filename)
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("実行情報を保存できませんでした。再試行できます。", exc)
            return
        self.record_info_var.set(f"実行情報: 保存済み / {metadata['run_id']} / {filename}")

    def save_research_record(self):
        capture = self.research_capture
        if self._running or capture is None or self.result_origin not in ("bl", "ga"):
            return
        initial = default_research_directory()
        initialdir = initial if initial.is_dir() else initial.parent
        selected = filedialog.askdirectory(
            title="研究記録の保存先を選択", initialdir=str(initialdir), mustexist=True
        )
        if not selected:
            return
        try:
            output_root = validate_research_output_root(selected)
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("研究記録の保存先が不正です。", exc)
            return
        save_level = "summary"
        if self.ga_result is not None:
            full = messagebox.askyesnocancel(
                "保存レベル",
                "Fullを選ぶと全世代・全個体の配置とPP入力状態を保存します。\n"
                "Summaryだけを保存する場合は「いいえ」を選んでください。",
                parent=self.root,
            )
            if full is None:
                return
            save_level = "full" if full else "summary"
        record_id = str(uuid4())
        destination = output_root / "records" / record_id
        try:
            saved = save_research_record(
                destination,
                capture,
                result=self.result if self.ga_result is None else None,
                run=self.ga_result,
                save_level=save_level,
                record_id=record_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            self.status_var.set(f"研究記録を保存できませんでした: {destination}")
            self._show_error("研究記録を保存できませんでした。結果は未確定です。", exc)
            return
        self.record_info_var.set(f"研究記録: {saved['record_id']} / {saved['status']} / {destination}")
        self.status_var.set("配置結果を別の検査処理で確認し、研究記録を保存しました。")

    @staticmethod
    def _record_selection_label(metadata) -> str:
        decoder = metadata["decoder"]
        settings = (
            f"{decoder['candidate_policy']} v{decoder['candidate_policy_version']} / "
            f"支持{decoder['min_support_percent']}% ({decoder['support_model_version']})"
        )
        return (
            f"{metadata['run_id']} | {decoder['name']} | seed={metadata['ga']['seed']} | "
            f"workers={metadata['execution']['workers']['resolved_workers']} | {settings}"
        )

    def _select_record(self, records):
        if len(records) == 1:
            return records[0]
        if not records:
            raise ValueError("比較レポートにtrialがありません。")
        dialog = tk.Toplevel(self.root)
        dialog.title("再実行するtrialを選択")
        dialog.transient(self.root)
        selected = []
        listing = tk.Listbox(dialog, width=140, height=min(12, len(records)), exportselection=False)
        listing.pack(fill="both", expand=True, padx=12, pady=12)
        for m in records:
            listing.insert("end", self._record_selection_label(m))
        def choose():
            index = listing.curselection()
            if index:
                selected.append(records[index[0]])
                dialog.destroy()
        ttk.Button(dialog, text="読み込む", command=choose).pack(side="left", padx=12, pady=12)
        ttk.Button(dialog, text="取消", command=dialog.destroy).pack(side="right", padx=12, pady=12)
        dialog.grab_set()
        self.root.wait_window(dialog)
        return selected[0] if selected else None

    def open_run_metadata(self):
        if self._running:
            return
        filename = filedialog.askopenfilename(title="実行情報から設定を読み込む",
            initialdir=str(self.source_path.parent if self.source_path else PROJECT_ROOT),
            filetypes=(("JSONファイル", "*.json"), ("すべてのファイル", "*.*")))
        if not filename:
            return
        try:
            metadata = self._select_record(read_execution_records(filename))
            if metadata is None:
                return
            settings = prepare_manual_replay(metadata, origin="gui")
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("実行情報を読み込めませんでした。", exc)
            return
        # Every value and current executor limit has been checked before mutation.
        self._applying_replay = True
        try:
            self._clear_result()
            self.container, self.boxes = settings.container, settings.boxes
            self.source_path = None
            self.source_bytes = None
            self.fixture_sha256 = settings.options.fixture_sha256
            self.priority_var.set(settings.priority.value)
            self.ga_decoder_var.set(settings.decoder)
            self.candidate_policy_var.set(settings.decoder_config.candidate_policy)
            self.min_support_percent_var.set(str(settings.decoder_config.min_support_percent))
            self.ga_population_var.set(str(settings.config.population_size))
            self.ga_generations_var.set(str(settings.config.generation_count))
            self.ga_crossover_var.set(str(settings.config.crossover_rate))
            self.ga_mutation_var.set(str(settings.config.mutation_rate))
            self.ga_cog_var.set(settings.config.use_center_of_gravity)
            self.ga_cog_lambda_var.set(str(settings.config.center_of_gravity_lambda))
            self.ga_seed_var.set(str(settings.seed))
            self.ga_workers_mode_var.set("Manual")
            self.ga_workers_var.set(str(settings.execution.workers))
            self.ga_cache_var.set(settings.execution.cache)
            self.source_var.set(f"実行情報: {filename}")
            self.container_var.set(f"{settings.container.W} × {settings.container.H} × {settings.container.D}")
            self.box_count_var.set(str(len(settings.boxes)))
            self.pack_button.state(["!disabled"])
            self.ga_button.state(["!disabled"])
            self.experiment_button.state(["!disabled"])
        finally:
            self._applying_replay = False
        self._on_settings_changed()
        self.replay_settings = settings
        self.problem_options = replace(settings.options, request=None, replay_of=None)
        self.workers_info_var.set(self.workers_info_var.get() + f" / 再現元={metadata['execution']['workers']['requested_mode']}")
        self.status_var.set("Manual再実行の設定を読み込みました。未実行。環境差: " +
                            ("; ".join(settings.environment_differences) or "検出なし"))

    def _reset_ga_progress(self) -> None:
        self._ga_started_at = None
        self._last_ga_progress = None
        self.ga_progress_percent_var.set(0.0)
        self.ga_progress_var.set("GA進捗: —")

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _apply_ga_progress(self, progress: GAProgress) -> None:
        self._last_ga_progress = progress
        percent = progress.fraction * 100.0
        self.ga_progress_percent_var.set(percent)
        elapsed = 0.0 if self._ga_started_at is None else time.monotonic() - self._ga_started_at
        self.ga_progress_var.set(
            f"GA進捗: {progress.completed_evaluations}/{progress.total_evaluations} "
            f"({percent:.1f}%) | 世代 {progress.generation_index + 1}/{progress.generation_count} "
            f"| 個体 {progress.population_index + 1}/{progress.population_size} "
            f"| {progress.decoder_name} | 経過 {self._format_elapsed(elapsed)}"
        )

    def _set_result_actions(self, enabled: bool) -> None:
        buttons = (
            self.validate_button,
            self.save_json_button,
            self.save_unity_button,
            self.copy_unity_button,
        )
        for button in buttons:
            button.state(["!disabled"] if enabled else ["disabled"])

    def _clear_result(self) -> None:
        self.result = None
        self.result_origin = None
        self.result_expected_boxes = None
        self.result_decoder_config = None
        self.ga_result = None
        self.research_capture = None
        self.record_state.clear()
        self.record_info_var.set("実行情報: —")
        self.save_metadata_button.state(["disabled"])
        self.research_save_button.state(["disabled"])
        self.ga_info_var.set("実行情報: —")
        self.placed_var.set("—")
        self.failed_var.set("—")
        self.volume_var.set("—")
        self.density_var.set("—")
        self.validation_var.set("未検証")
        self._reset_ga_progress()
        self._set_result_actions(False)
        self.tree.delete(*self.tree.get_children())
        self._set_text(self.json_preview, "")
        self._set_text(self.unity_preview, "")

    def _load_box_sizes(self, path: Path, source_label: str) -> None:
        try:
            raw = path.read_bytes()
            container, boxes = parse_box_sizes_bytes(raw, source=str(path))
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("入力ファイルを読み込めませんでした。", exc)
            return

        self.container = container
        self.boxes = boxes
        self.replay_settings = None
        self.fixture_sha256 = hashlib.sha256(raw).hexdigest()
        self.problem_options = RunOptions(origin="gui", source=str(path), fixture_sha256=self.fixture_sha256)
        self.source_path = path
        self._clear_result()
        self.source_bytes = raw
        self.source_var.set(source_label)
        self.container_var.set(f"{container.W} × {container.H} × {container.D}")
        self.box_count_var.set(str(len(boxes)))
        self.pack_button.state(["!disabled"])
        self.ga_button.state(["!disabled"])
        self.experiment_button.state(["!disabled"])
        self.status_var.set("入力を読み込みました。GAまたはBL単体を実行できます。")

    def open_box_sizes(self) -> None:
        filename = filedialog.askopenfilename(
            title="box_sizes.txt を選択",
            initialdir=str(self.source_path.parent if self.source_path else PROJECT_ROOT),
            filetypes=(("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")),
        )
        if not filename:
            return
        path = Path(filename)
        self._load_box_sizes(path, str(path))

    def use_sample(self) -> None:
        self._load_box_sizes(SAMPLE_PATH, f"付属サンプル: {SAMPLE_PATH.name}")

    def open_result_json(self) -> None:
        filename = filedialog.askopenfilename(
            title="結果JSONを開く",
            initialdir=str(self.source_path.parent if self.source_path else PROJECT_ROOT),
            filetypes=(("JSONファイル", "*.json"), ("すべてのファイル", "*.*")),
        )
        if not filename:
            return

        path = Path(filename)
        try:
            result = read_result_json(path)
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("結果JSONを読み込めませんでした。", exc)
            return

        self.container = None
        self.boxes = None
        self.replay_settings = None
        self.fixture_sha256 = None
        self.source_bytes = None
        self.record_state.clear()
        self.record_info_var.set("実行情報: —")
        self.save_metadata_button.state(["disabled"])
        self.research_capture = None
        self.research_save_button.state(["disabled"])
        self.ga_result = None
        self.source_path = path
        self.source_var.set(f"結果JSON: {path}")
        self.container_var.set(f"{result.container.W} × {result.container.H} × {result.container.D}")
        self.box_count_var.set(str(len(result.placements)))
        self.pack_button.state(["disabled"])
        self.ga_button.state(["disabled"])
        self.experiment_button.state(["disabled"])
        self._show_result(
            result,
            origin="json",
            expected_boxes=None,
            run_info="実行情報: 保存済み結果JSON",
        )
        self.status_var.set("結果JSONを読み込みました。生成条件が不明のため支持は未検査です。")

    def _begin_run(self, status: str) -> bool:
        if self._running or self.container is None or self.boxes is None:
            return False
        self._clear_result()
        self._running = True
        self._set_input_actions(False)
        self.pack_button.state(["disabled"])
        self.ga_button.state(["disabled"])
        self.status_var.set(status)
        return True

    def start_pack(self) -> None:
        if self._running or self.container is None or self.boxes is None:
            return
        try:
            decoder_config = parse_decoder_settings(
                self.candidate_policy_var.get(), self.min_support_percent_var.get()
            )
        except (TypeError, ValueError) as exc:
            self._show_error("decoder設定が不正です。", exc)
            return
        container = self.container
        boxes = self.boxes
        assert container is not None and boxes is not None
        priority = BLPriority(self.priority_var.get())
        options = self.problem_options
        snapshot = snapshot_for_single(
            container,
            boxes,
            record_type="BL",
            priority=priority,
            decoder_config=decoder_config,
            source=options.source,
            fixture_sha256=options.fixture_sha256,
            order_mode=options.order_mode,
            source_boxes=options.source_boxes,
        )
        execution_started_at = utc_now()
        execution_provenance = make_provenance()
        source_bytes = self.source_bytes
        if not self._begin_run(f"BL配置中... ({self.priority_var.get()})"):
            return

        def worker() -> None:
            try:
                call_started = time.perf_counter()
                result = pack(
                    container,
                    boxes,
                    priority=priority,
                    decoder_config=decoder_config,
                )
                call_wall_seconds = time.perf_counter() - call_started
            except Exception as exc:  # Surface unexpected API failures to the user.
                self._pack_queue.put(("error_bl", exc))
            else:
                capture = capture_single_run(
                    snapshot=snapshot,
                    execution_started_at=execution_started_at,
                    execution_finished_at=utc_now(),
                    call_wall_seconds=call_wall_seconds,
                    provenance=execution_provenance,
                    source_bytes=source_bytes,
                )
                self._pack_queue.put(("ok_bl", (result, priority, decoder_config, capture)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll_pack_queue)

    def start_ga(self) -> None:
        if self._running or self.container is None or self.boxes is None:
            return
        try:
            config = GAConfig(
                population_size=int(self.ga_population_var.get()),
                generation_count=int(self.ga_generations_var.get()),
                crossover_rate=float(self.ga_crossover_var.get()),
                mutation_rate=float(self.ga_mutation_var.get()),
                use_center_of_gravity=bool(self.ga_cog_var.get()),
                center_of_gravity_lambda=float(self.ga_cog_lambda_var.get()),
            )
            seed = int(self.ga_seed_var.get())
            if seed < 0:
                raise ValueError("seed must be >= 0")
            priority = BLPriority(self.priority_var.get())
            decoder = self.ga_decoder_var.get()
            if decoder not in ("BL", "PP"):
                raise ValueError("decoder must be BL or PP")
            decoder_config = (
                self.replay_settings.decoder_config
                if self.replay_settings is not None
                else parse_decoder_settings(
                    self.candidate_policy_var.get(), self.min_support_percent_var.get()
                )
            )
            request = parse_workers_request(self.ga_workers_mode_var.get(), self.ga_workers_var.get())
            execution = GAExecutionConfig(workers=1, cache=bool(self.ga_cache_var.get()))
            options = (self.replay_settings.options if self.replay_settings is not None else
                       self.problem_options)
            options = replace(options, request=request,
                              resolution_callback=lambda ex, env: self._pack_queue.put(("resolved_ga", (ex, env))))
        except (TypeError, ValueError) as exc:
            self._show_error("GA設定が不正です。", exc)
            return

        execution_started_at = utc_now()
        execution_provenance = make_provenance()
        source_bytes = self.source_bytes

        if not self._begin_run(
            f"GA実行中... decoder={decoder}, 個体数={config.population_size}, "
            f"世代数={config.generation_count}, priority={priority.value}, "
            f"policy={decoder_config.candidate_policy}, support={decoder_config.min_support_percent}%, seed={seed}, "
            f"workers要求={request.mode}, cache={'ON' if execution.cache else 'OFF'}"
        ):
            return

        total_evaluations = config.population_size * config.generation_count
        self._ga_started_at = time.monotonic()
        self.ga_progress_percent_var.set(0.0)
        self.ga_progress_var.set(
            f"GA進捗: 0/{total_evaluations} (0.0%) | 世代 0/{config.generation_count} "
            f"| {decoder} | 経過 00:00"
        )

        container = self.container
        boxes = self.boxes
        assert container is not None and boxes is not None

        def worker() -> None:
            try:
                call_started = time.perf_counter()
                run = run_ga_core(
                    container,
                    boxes,
                    decoder=decoder,
                    priority=priority,
                    config=config,
                    seed=seed,
                    execution=execution,
                    decoder_config=decoder_config,
                    run_options=options,
                    progress_callback=lambda progress: self._pack_queue.put(("progress_ga", progress)),
                )
                capture = capture_ga_run(
                    run,
                    execution_started_at=execution_started_at,
                    execution_finished_at=utc_now(),
                    call_wall_seconds=time.perf_counter() - call_started,
                    source_bytes=source_bytes,
                    execution_provenance=execution_provenance,
                )
            except Exception as exc:
                self._pack_queue.put(("error_ga", exc))
            else:
                self._pack_queue.put(("ok_ga", (run, capture)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll_pack_queue)

    def _poll_pack_queue(self) -> None:
        latest_progress: GAProgress | None = None
        terminal: tuple[str, object] | None = None

        while True:
            try:
                kind, payload = self._pack_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "resolved_ga":
                ex, env = payload
                self.workers_info_var.set(self._workers_description(ex, env, "確定"))
                continue
            if kind == "progress_ga":
                if isinstance(payload, GAProgress):
                    latest_progress = payload
                continue
            terminal = (kind, payload)

        if latest_progress is not None:
            self._apply_ga_progress(latest_progress)

        if terminal is None:
            if self._running:
                self.root.after(50, self._poll_pack_queue)
            return

        kind, payload = terminal
        self._running = False
        self._set_input_actions(True)
        if self.container is not None and self.boxes is not None:
            self.pack_button.state(["!disabled"])
            self.ga_button.state(["!disabled"])

        if kind.startswith("error_"):
            assert isinstance(payload, Exception)
            if kind == "error_ga":
                elapsed = 0.0 if self._ga_started_at is None else time.monotonic() - self._ga_started_at
                if self._last_ga_progress is None:
                    self.ga_progress_var.set(f"GA進捗: 失敗 | 経過 {self._format_elapsed(elapsed)}")
                else:
                    self.ga_progress_var.set(self.ga_progress_var.get() + " | 失敗")
                self.status_var.set("GA実行に失敗しました。")
                self._show_error("GAを実行できませんでした。", payload)
            else:
                self.status_var.set("BL配置に失敗しました。")
                self._show_error("BL配置を実行できませんでした。", payload)
            return

        if kind == "ok_bl":
            result, priority, decoder_config, capture = payload
            assert isinstance(result, PackingResult)
            assert isinstance(priority, BLPriority)
            assert isinstance(decoder_config, DecoderConfig)
            self.research_capture = capture
            self.research_save_button.state(["!disabled"])
            expected_boxes = self.boxes
            self._show_result(
                result,
                origin="bl",
                expected_boxes=expected_boxes,
                decoder_config=decoder_config,
                run_info=(
                    f"実行情報: BL単体 / {priority.value} / "
                    f"{decoder_config.candidate_policy} v{decoder_config.candidate_policy_version} / "
                    f"支持{decoder_config.min_support_percent}%"
                ),
            )
            self.status_var.set("BL配置が完了しました。")
            return

        assert kind == "ok_ga" and isinstance(payload, tuple) and len(payload) == 2
        run, capture = payload
        assert isinstance(run, (GARunResult, PPGARunResult))
        self.research_capture = capture
        self.research_save_button.state(["!disabled"])
        self.ga_result = run
        best = run.best
        by_id = {box.id: box for box in run.boxes}
        expected_boxes = tuple(by_id[box_id] for box_id in best.individual.genes)
        self._show_result(
            best.packing_result,
            origin="ga",
            expected_boxes=expected_boxes,
            run_info=(
                f"実行情報: GA({run.decoder_name}) / {run.priority.value} / "
                f"{run.decoder_config.candidate_policy} v{run.decoder_config.candidate_policy_version} / "
                f"支持{run.decoder_config.min_support_percent}% / seed={run.seed} / "
                f"workers={run.execution.workers}, cache={'ON' if run.execution.cache else 'OFF'} / "
                f"評価={run.execution_stats.requested}, 実計算={run.execution_stats.computed}, "
                f"再利用={run.execution_stats.cache_hits} / "
                f"best=世代{run.best_generation_index + 1}, 個体{run.best_population_index + 1}, "
                f"fitness={best.fitness:.6f}"
            ),
            decoder_config=run.decoder_config,
        )
        elapsed = 0.0 if self._ga_started_at is None else time.monotonic() - self._ga_started_at
        self.ga_progress_percent_var.set(100.0)
        self.ga_progress_var.set(
            f"GA進捗: {run.execution_stats.requested}/{run.execution_stats.requested} (100.0%) "
            f"| 完了 | 経過 {self._format_elapsed(elapsed)}"
        )
        self.ga_result = run
        self.record_state.complete(run)
        self.record_info_var.set(f"実行情報: 未保存 / run_id={run.run_metadata['run_id']}")
        self.save_metadata_button.state(["!disabled"])
        self.workers_info_var.set(self._workers_description(run.run_metadata['execution'], run.run_metadata['environment'], "確定"))
        self.status_var.set("GA実行が完了しました。最良個体の配置を表示しています。")

    def _show_result(
        self,
        result: PackingResult,
        *,
        origin: str,
        expected_boxes,
        run_info: str,
        decoder_config: DecoderConfig | None = None,
    ) -> None:
        self.result = result
        self.result_origin = origin
        self.result_expected_boxes = expected_boxes
        self.result_decoder_config = decoder_config
        self.ga_info_var.set(run_info)
        self.placed_var.set(str(result.placed_count))
        self.failed_var.set(str(result.failed_count))
        self.volume_var.set(str(result.placed_volume))
        self.density_var.set(f"{result.density * 100:.2f} %")
        self.validation_var.set(
            "未検証"
            if decoder_config is not None
            else "未検証 / 支持未検査"
        )
        self._set_result_actions(True)

        self.tree.delete(*self.tree.get_children())
        for placement in result.placements:
            box = placement.box
            size = f"{box.w} × {box.h} × {box.d}"
            if placement.position is None:
                position = "—"
                state = "未配置"
            else:
                p = placement.position
                position = f"({p.x}, {p.y}, {p.z})"
                state = "配置済み"
            self.tree.insert("", "end", values=(box.id, size, position, state))

        self._refresh_previews()

    def validate_current_result(self) -> None:
        if self.result is None:
            return
        try:
            if self.container is not None and self.result_expected_boxes is not None:
                geometry_errors = validate_result(
                    self.result,
                    expected_container=self.container,
                    ordered_boxes=self.result_expected_boxes,
                )
            else:
                geometry_errors = validate_result(self.result)
            support_errors = ()
            if not geometry_errors and self.result_decoder_config is not None:
                support_errors = validate_support(
                    self.result,
                    settings=self.result_decoder_config,
                )
        except (TypeError, ValueError) as exc:
            self._show_error("結果を検証できませんでした。", exc)
            return

        if geometry_errors:
            self.validation_var.set(
                f"幾何学エラー {len(geometry_errors)} 件 / 支持未検査"
            )
            self.status_var.set(f"結果検証: 幾何学上の問題が{len(geometry_errors)}件見つかりました。")
            messagebox.showerror(
                "結果検証",
                f"幾何学上の問題が{len(geometry_errors)}件見つかりました。\n\n"
                + "\n".join(geometry_errors),
                parent=self.root,
            )
        elif support_errors:
            self.validation_var.set(f"支持エラー {len(support_errors)} 件")
            self.status_var.set(f"結果検証: 支持条件の問題が{len(support_errors)}件見つかりました。")
            messagebox.showerror(
                "結果検証",
                f"支持条件の問題が{len(support_errors)}件見つかりました。\n\n"
                + "\n".join(support_errors),
                parent=self.root,
            )
        elif self.result_decoder_config is None:
            self.validation_var.set("配置範囲・重なり: 問題なし / 支持条件: 未確認")
            self.status_var.set("結果検証: 幾何学OK。生成条件が不明のため支持は未検査です。")
            messagebox.showinfo(
                "結果検証",
                "幾何学検証OK\n\n支持条件は実行時の設定が不明なため確認できません。",
                parent=self.root,
            )
        else:
            settings = self.result_decoder_config
            self.validation_var.set(
                f"検証OK / 支持{settings.min_support_percent}%確認"
            )
            self.status_var.set("結果検証: 幾何学・設定済み支持条件ともに問題は見つかりませんでした。")
            messagebox.showinfo(
                "結果検証",
                f"幾何学検証OK\n支持条件: {settings.min_support_percent}% / "
                f"{settings.support_model_version}\n\n問題は見つかりませんでした。",
                parent=self.root,
            )

    def save_result_json(self) -> None:
        if self.result is None:
            return
        filename = filedialog.asksaveasfilename(
            title="結果JSONを保存",
            initialdir=str(self.source_path.parent if self.source_path else PROJECT_ROOT),
            initialfile="result.json",
            defaultextension=".json",
            filetypes=(("JSONファイル", "*.json"), ("すべてのファイル", "*.*")),
        )
        if not filename:
            return
        try:
            write_result_json(filename, self.result)
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("結果JSONを保存できませんでした。", exc)
            return
        self.status_var.set(f"結果JSONを保存しました: {filename}")

    def save_unity_data(self) -> None:
        if self.result is None:
            return
        filename = filedialog.asksaveasfilename(
            title="Unity用データを保存",
            initialdir=str(self.source_path.parent if self.source_path else PROJECT_ROOT),
            initialfile="ResultData.txt",
            defaultextension=".txt",
            filetypes=(("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")),
        )
        if not filename:
            return
        try:
            write_unity_result_data(filename, self.result)
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("Unity用データを保存できませんでした。", exc)
            return
        self.status_var.set(f"Unity用データを保存しました: {filename}")

    def copy_unity_data(self) -> None:
        if self.result is None:
            return
        try:
            with tempfile.TemporaryDirectory(prefix="packing3d_gui_") as tmp:
                unity_path = Path(tmp) / "ResultData.txt"
                write_unity_result_data(unity_path, self.result)
                unity_text = unity_path.read_text(encoding="utf-8").rstrip("\n")
            self.root.clipboard_clear()
            self.root.clipboard_append(unity_text)
            self.root.update_idletasks()
        except (OSError, TypeError, ValueError, tk.TclError) as exc:
            self._show_error("Unity形式をコピーできませんでした。", exc)
            return
        self.status_var.set("現在の配置結果をUnity形式でコピーしました。")

    def _refresh_previews(self) -> None:
        if self.result is None:
            self._set_text(self.json_preview, "")
            self._set_text(self.unity_preview, "")
            return

        try:
            json_text, unity_text = render_previews(self.result)
        except (OSError, TypeError, ValueError) as exc:
            self._set_text(self.json_preview, f"プレビューを生成できませんでした。\n\n{exc}")
            self._set_text(self.unity_preview, f"プレビューを生成できませんでした。\n\n{exc}")
            return

        self._set_text(self.json_preview, json_text)
        self._set_text(self.unity_preview, unity_text)

    def _show_error(self, title: str, exc: Exception) -> None:
        messagebox.showerror(
            "エラー",
            f"{title}\n\n詳細:\n{exc}",
            parent=self.root,
        )
        self.status_var.set(title)


def _notify_launcher_ready() -> None:
    ready_file = os.environ.get("PACKING3D_GUI_READY_FILE")
    if ready_file:
        Path(ready_file).write_text("ready\n", encoding="ascii")


def main() -> None:
    root = tk.Tk()
    try:
        root.option_add("*Font", ("Segoe UI", 10))
    except tk.TclError:
        pass
    PackingDemoApp(root)
    _notify_launcher_ready()
    root.mainloop()


if __name__ == "__main__":
    main()
