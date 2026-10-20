"""Dedicated Tk dialog for sequential Research Experiments."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from packing import GAConfig, BLPriority
from packing.research_records import default_research_directory
from packing.research_experiment import prepare_research_experiment, run_research_experiment
from gui_core import parse_decoder_settings, parse_workers_request


class ResearchExperimentDialog:
    def __init__(self, app) -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Research Experiment")
        self.window.transient(app.root)
        self.window.protocol("WM_DELETE_WINDOW", self.request_close)
        self.window.geometry("760x440")
        self.stop_event = threading.Event()
        self.messages: queue.Queue[dict] = queue.Queue()
        self.plan = None
        self.worker = None
        self.closing = False
        self.started_monotonic = None
        self.trial_states: list[str] = []

        self.count_var = tk.StringVar(value="1")
        self.seed_var = tk.StringVar(value=app.ga_seed_var.get())
        self.level_var = tk.StringVar(value="summary")
        self.profile_var = tk.BooleanVar(value=False)
        self.output_var = tk.StringVar(value=str(default_research_directory()))
        self.status_var = tk.StringVar(value="条件を確認してExperimentを開始できます。")
        self.preview_var = tk.StringVar(value="")
        self.elapsed_var = tk.StringVar(value="経過: —")

        frame = ttk.Frame(self.window, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="実行数 R:").grid(row=0, column=0, sticky="w")
        self.count_entry = ttk.Entry(frame, textvariable=self.count_var, width=10)
        self.count_entry.grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="開始seed:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.seed_entry = ttk.Entry(frame, textvariable=self.seed_var, width=14)
        self.seed_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(frame, text="保存レベル:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.level_combo = ttk.Combobox(frame, textvariable=self.level_var, values=("summary", "full"),
                                        state="readonly", width=12)
        self.level_combo.grid(row=2, column=1, sticky="w", pady=(8, 0))
        self.profile_check = ttk.Checkbutton(frame, text="Profileを計測", variable=self.profile_var)
        self.profile_check.grid(
            row=3, column=1, sticky="w", pady=(8, 0))
        ttk.Label(frame, text="保存先:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.output_entry = ttk.Entry(frame, textvariable=self.output_var)
        self.output_entry.grid(row=4, column=1, sticky="ew", pady=(8, 0))
        self.browse_button = ttk.Button(frame, text="選択…", command=self.choose_output)
        self.browse_button.grid(row=4, column=2, padx=(8, 0), pady=(8, 0))
        ttk.Label(frame, textvariable=self.preview_var, justify="left", wraplength=700).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(16, 0))
        ttk.Label(frame, textvariable=self.status_var, wraplength=700).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Label(frame, textvariable=self.elapsed_var).grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.trial_tree = ttk.Treeview(frame, columns=("index", "seed", "status"), show="headings", height=7)
        for key, title, width in (("index", "試行", 70), ("seed", "seed", 150), ("status", "状態", 130)):
            self.trial_tree.heading(key, text=title)
            self.trial_tree.column(key, width=width, anchor="center")
        self.trial_tree.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        frame.rowconfigure(8, weight=1)
        actions = ttk.Frame(frame)
        actions.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(18, 0))
        self.start_button = ttk.Button(actions, text="条件を確認して開始", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="現在の試行完了後に停止", command=self.request_stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        self.stop_button.state(["disabled"])
        ttk.Button(actions, text="閉じる", command=self.request_close).pack(side="right")
        self.window.after(100, self.poll)

    def choose_output(self) -> None:
        selected = filedialog.askdirectory(
            title="Experiment保存先", initialdir=self.output_var.get(), mustexist=False, parent=self.window)
        if selected:
            self.output_var.set(selected)

    def _build_plan(self):
        app = self.app
        if app.container is None or app.boxes is None:
            raise ValueError("先にbox_sizes入力を読み込んでください")
        config = GAConfig(
            population_size=int(app.ga_population_var.get()),
            generation_count=int(app.ga_generations_var.get()),
            crossover_rate=float(app.ga_crossover_var.get()),
            mutation_rate=float(app.ga_mutation_var.get()),
            use_center_of_gravity=bool(app.ga_cog_var.get()),
            center_of_gravity_lambda=float(app.ga_cog_lambda_var.get()),
        )
        request = parse_workers_request(app.ga_workers_mode_var.get(), app.ga_workers_var.get())
        decoder = parse_decoder_settings(app.candidate_policy_var.get(), app.min_support_percent_var.get())
        options = app.replay_settings.options if app.replay_settings is not None else app.problem_options
        options = type(options)(
            origin=options.origin, source_boxes=options.source_boxes, order_mode=options.order_mode,
            source=options.source, fixture_sha256=options.fixture_sha256,
            implementation_version=options.implementation_version, revision=options.revision,
        )
        plan = prepare_research_experiment(
            app.container, app.boxes, record_type=f"{app.ga_decoder_var.get()}-GA", config=config,
            priority=BLPriority(app.priority_var.get()), decoder_config=decoder,
            start_seed=int(self.seed_var.get()), run_count=int(self.count_var.get()),
            workers_request=request, cache=bool(app.ga_cache_var.get()), save_level=self.level_var.get(),
            profile=bool(self.profile_var.get()), run_options=options, source_bytes=app.source_bytes,
        )
        return plan

    def start(self) -> None:
        if self.worker is not None:
            return
        try:
            self.plan = self._build_plan()
            directory = self.output_var.get().strip()
            if not directory:
                raise ValueError("保存先を指定してください")
        except (TypeError, ValueError, OSError) as exc:
            messagebox.showerror("Experiment条件を確認できません", str(exc), parent=self.window)
            return
        plan = self.plan
        evaluations = plan.run_count * plan.config.population_size * plan.config.generation_count
        box_count = len(plan.boxes)
        summary = (
            f"対象: {plan.record_type} / {plan.fixed_snapshot['decoder']['priority']} / "
            f"{plan.fixed_snapshot['decoder']['candidate_policy']} / "
            f"支持{plan.fixed_snapshot['decoder']['min_support_percent']}%\n"
            f"P={plan.config.population_size}, G={plan.config.generation_count}, "
            f"R={plan.run_count}, seed={plan.start_seed}〜{plan.start_seed + plan.run_count - 1}\n"
            f"workers: {plan.workers_selection['request']['mode']} → Manual({plan.workers}), "
            f"cache={'ON' if plan.cache else 'OFF'}, profile={'ON' if plan.profile else 'OFF'}, "
            f"level={plan.save_level}\n"
            f"容量注意: {evaluations}評価・{box_count}箱。Fullは履歴artifactが増えます。"
            "Summaryでもcoreは1run分の全履歴を保持します。容量の保証値ではありません。\n"
            f"保存先: {directory}"
        )
        self.preview_var.set(summary)
        if not messagebox.askyesno(
            "Experiment条件の確認", summary + "\n\nこの固定条件で開始しますか？",
            parent=self.window,
        ):
            return
        self.stop_event.clear()
        self.app._research_experiment_active = True
        self.app._set_input_actions(False)
        self.app.pack_button.state(["disabled"])
        self.app.ga_button.state(["disabled"])
        for widget in (self.count_entry, self.seed_entry, self.level_combo, self.profile_check, self.output_entry):
            widget.configure(state="disabled")
        self.start_button.state(["disabled"])
        self.browse_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.trial_tree.delete(*self.trial_tree.get_children())
        self.trial_states = ["planned"] * plan.run_count
        for index in range(plan.run_count):
            self.trial_tree.insert("", "end", iid=str(index),
                                   values=(index + 1, plan.start_seed + index, "未実行"))
        self.started_monotonic = time.monotonic()
        self.status_var.set("Experiment実行中です。停止要求後も現在の試行の検査・保存まで待機します。")
        self.worker = threading.Thread(
            target=self._run_worker, args=(plan, directory), name="packing3d-research-experiment", daemon=False)
        self.worker.start()

    def _run_worker(self, plan, directory: str) -> None:
        try:
            result = run_research_experiment(
                plan, directory, stop_event=self.stop_event,
                progress_callback=self.messages.put,
            )
            self.messages.put({"kind": "terminal", "result": result, "directory": str(directory)})
        except Exception as exc:
            self.messages.put({"kind": "worker_error", "error": exc})

    def request_stop(self) -> None:
        if self.worker is None:
            self.destroy()
            return
        self.stop_event.set()
        self.status_var.set("停止要求を受け付けました。現在の試行完了後に停止します。")
        self.stop_button.state(["disabled"])
        self.closing = self.closing or False

    def request_close(self) -> None:
        if self.worker is None:
            self.destroy()
            return
        self.closing = True
        self.request_stop()
        self.status_var.set("閉じる要求を受け付けました。現在の試行完了後に停止してから閉じます。")

    def poll(self) -> None:
        terminal = None
        if self.worker is not None and self.started_monotonic is not None:
            self.elapsed_var.set(f"経過: {time.monotonic() - self.started_monotonic:.1f} 秒")
        while True:
            try:
                item = self.messages.get_nowait()
            except queue.Empty:
                break
            kind = item["kind"]
            if kind == "trial_completed":
                self.trial_states[item["run_index"]] = "completed"
                self.trial_tree.item(str(item["run_index"]), values=(
                    item["run_index"] + 1, item["seed"], "完了"))
                self.status_var.set(
                    f"試行 {item['run_index'] + 1}/{self.plan.run_count} 完了・保存済み / "
                    f"seed={item['seed']} / fitness={item['fitness']:.6g} / run={item['run_id']}"
                )
            elif kind == "progress":
                if self.trial_states[item["run_index"]] not in ("completed", "failed"):
                    self.trial_states[item["run_index"]] = "running"
                    self.trial_tree.item(str(item["run_index"]), values=(
                        item["run_index"] + 1, self.plan.start_seed + item["run_index"], "実行中"))
                self.status_var.set(
                    f"試行 {item['run_index'] + 1}/{self.plan.run_count}: "
                    f"世代 {item['generation_index'] + 1}/{item['generation_count']}, "
                    f"個体 {item['population_index'] + 1}/{item['population_size']} "
                    f"({item['decoder']})"
                )
            elif kind == "warning":
                self.status_var.set(f"Experimentは継続しますが、summary.csvを更新できません: {item['message']}")
            elif kind == "failed":
                self.trial_states[item["run_index"]] = "failed"
                self.trial_tree.item(str(item["run_index"]), values=(
                    item["run_index"] + 1, self.plan.start_seed + item["run_index"], "失敗"))
            elif kind in ("terminal", "worker_error"):
                terminal = item
        if terminal is not None:
            self.worker.join()
            if terminal["kind"] == "terminal":
                result = terminal["result"]
                for index, state in enumerate(self.trial_states):
                    if state == "planned" and result["status"] == "stopped":
                        self.trial_states[index] = "planned"
                        self.trial_tree.item(str(index), values=(
                            index + 1, self.plan.start_seed + index, "未実行"))
                completed = self.trial_states.count("completed")
                failed = self.trial_states.count("failed")
                planned = self.trial_states.count("planned")
                summary = (f"Experiment {result['status']}: 完了{completed} / 失敗{failed} / "
                           f"未実行{planned} / {terminal['directory']}")
                self.status_var.set(summary)
                self.app.status_var.set(summary)
            else:
                error = terminal["error"]
                self.status_var.set(f"Experimentの記録を保存できませんでした: {error}")
                self.app.status_var.set("Experimentに失敗しました。詳細は専用画面を確認してください。")
            self.app._research_experiment_active = False
            self.app._set_input_actions(True)
            if self.app.container is not None and self.app.boxes is not None:
                self.app.pack_button.state(["!disabled"])
                self.app.ga_button.state(["!disabled"])
            if self.closing or getattr(self.app, "_close_when_experiment_done", False):
                self.destroy()
                if getattr(self.app, "_close_when_experiment_done", False):
                    self.app.root.destroy()
                return
            self.start_button.state(["!disabled"])
            self.browse_button.state(["!disabled"])
            self.stop_button.state(["disabled"])
            self.worker = None
            self.window.after(100, self.poll)
            return
        self.window.after(100, self.poll)

    def destroy(self) -> None:
        self.app._research_experiment_dialog = None
        self.app._research_experiment_active = False
        try:
            self.window.destroy()
        except tk.TclError:
            pass
