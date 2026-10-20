from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import research_gui


class _Var:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _Widget:
    def __init__(self):
        self.states = set()
        self.rows = {}
        self.callbacks = []

    def state(self, values):
        for value in values:
            if value.startswith("!"):
                self.states.discard(value[1:])
            else:
                self.states.add(value)

    def configure(self, **values):
        self.config = values

    def delete(self, *_items):
        self.rows.clear()

    def insert(self, _parent, _index, *, iid, values):
        self.rows[iid] = values

    def get_children(self):
        return tuple(self.rows)

    def item(self, item, *, values):
        self.rows[item] = values

    def after(self, delay, callback):
        self.callbacks.append((delay, callback))

    def destroy(self):
        self.destroyed = True


class _Worker:
    def __init__(self):
        self.joined = False

    def join(self):
        self.joined = True


def test_close_requests_cooperative_stop_and_does_not_destroy_running_window():
    dialog = research_gui.ResearchExperimentDialog.__new__(research_gui.ResearchExperimentDialog)
    dialog.worker = _Worker()
    dialog.window = _Widget()
    dialog.stop_event = threading.Event()
    dialog.status_var = _Var()
    dialog.stop_button = _Widget()
    dialog.closing = False
    dialog.request_close()
    assert dialog.stop_event.is_set()
    assert dialog.closing
    assert not getattr(dialog.window, "destroyed", False)
    assert "disabled" in dialog.stop_button.states
    assert "停止" in dialog.status_var.value


def test_terminal_queue_restores_main_controls_and_reports_trial_counts():
    dialog = research_gui.ResearchExperimentDialog.__new__(research_gui.ResearchExperimentDialog)
    dialog.worker = _Worker()
    dialog.messages = queue.Queue()
    dialog.status_var = _Var()
    dialog.elapsed_var = _Var()
    dialog.started_monotonic = None
    dialog.plan = SimpleNamespace(run_count=2, start_seed=90)
    dialog.trial_states = ["planned", "planned"]
    dialog.trial_tree = _Widget()
    dialog.app = SimpleNamespace(
        _research_experiment_active=True, container=object(), boxes=object(),
        status_var=_Var(), pack_button=_Widget(), ga_button=_Widget(),
        _set_input_actions=lambda enabled: setattr(dialog.app, "enabled", enabled),
        _close_when_experiment_done=False,
        root=_Widget(),
    )
    dialog.window = _Widget()
    dialog.start_button = _Widget()
    dialog.browse_button = _Widget()
    dialog.stop_button = _Widget()
    dialog.closing = False
    dialog.messages.put({"kind": "progress", "run_index": 0, "generation_index": 1,
                         "generation_count": 3, "population_index": 2, "population_size": 4,
                         "decoder": "BL"})
    dialog.messages.put({"kind": "trial_completed", "run_index": 0, "seed": 90,
                         "record_id": "id", "run_id": "run", "fitness": 0.5})
    dialog.messages.put({"kind": "terminal", "result": {"status": "stopped"},
                         "directory": "C:/research/example"})
    dialog.poll()
    assert dialog.worker is None
    assert dialog.app.enabled is True
    assert dialog.app._research_experiment_active is False
    assert dialog.app.status_var.value == (
        "Experiment stopped: 完了1 / 失敗0 / 未実行1 / C:/research/example"
    )
    assert dialog.trial_tree.rows["0"] == (1, 90, "完了")
    assert dialog.trial_tree.rows["1"] == (2, 91, "未実行")
    assert dialog.start_button.states == set()


def test_root_close_waits_for_terminal_queue_before_destroying_toplevel_and_root():
    events = []

    class Worker:
        def join(self):
            events.append("join")

    dialog = research_gui.ResearchExperimentDialog.__new__(research_gui.ResearchExperimentDialog)
    dialog.worker = Worker()
    dialog.messages = queue.Queue()
    dialog.status_var = _Var()
    dialog.elapsed_var = _Var()
    dialog.started_monotonic = None
    dialog.plan = SimpleNamespace(run_count=1, start_seed=7)
    dialog.trial_states = ["completed"]
    dialog.trial_tree = _Widget()
    dialog.app = SimpleNamespace(
        _research_experiment_active=True, container=object(), boxes=object(),
        status_var=_Var(), pack_button=_Widget(), ga_button=_Widget(),
        _set_input_actions=lambda enabled: setattr(dialog.app, "enabled", enabled),
        _close_when_experiment_done=True, _research_experiment_dialog=dialog,
        root=SimpleNamespace(destroy=lambda: events.append("root")),
    )
    dialog.window = SimpleNamespace(destroy=lambda: events.append("window"))
    dialog.start_button = _Widget()
    dialog.browse_button = _Widget()
    dialog.stop_button = _Widget()
    dialog.closing = True
    dialog.messages.put({"kind": "terminal", "result": {"status": "completed"},
                         "directory": "C:/research/example"})

    dialog.poll()

    assert events == ["join", "window", "root"]
    assert dialog.app._research_experiment_dialog is None


def test_start_clears_a_stop_request_from_the_previous_run(monkeypatch):
    dialog = research_gui.ResearchExperimentDialog.__new__(research_gui.ResearchExperimentDialog)
    plan = SimpleNamespace(
        run_count=2,
        config=SimpleNamespace(population_size=2, generation_count=1),
        boxes=(object(), object()),
        fixed_snapshot={"decoder": {"priority": "BottomFirst", "candidate_policy": "Cartesian",
                                    "min_support_percent": 55}},
        record_type="BL-GA",
        workers_selection={"request": {"mode": "manual"}},
        workers=1,
        cache=False,
        profile=False,
        save_level="summary",
        start_seed=17,
    )
    dialog._build_plan = lambda: plan
    dialog.output_var = _Var("C:/research")
    dialog.preview_var = _Var()
    dialog.status_var = _Var()
    dialog.elapsed_var = _Var()
    dialog.count_entry = _Widget()
    dialog.seed_entry = _Widget()
    dialog.level_combo = _Widget()
    dialog.profile_check = _Widget()
    dialog.output_entry = _Widget()
    dialog.start_button = _Widget()
    dialog.browse_button = _Widget()
    dialog.stop_button = _Widget()
    dialog.trial_tree = _Widget()
    dialog.trial_states = []
    dialog.started_monotonic = None
    dialog.worker = None
    dialog.window = _Widget()
    dialog.stop_event = threading.Event()
    dialog.stop_event.set()
    app = SimpleNamespace(
        _research_experiment_active=False,
        container=object(),
        boxes=(object(),),
        pack_button=_Widget(),
        ga_button=_Widget(),
        _set_input_actions=lambda enabled: setattr(app, "input_enabled", enabled),
    )
    dialog.app = app
    started = []

    class _Thread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self)

    monkeypatch.setattr(research_gui.messagebox, "askyesno", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(research_gui.threading, "Thread", _Thread)
    dialog.start()

    assert started
    assert not dialog.stop_event.is_set()
    assert dialog.worker is started[0]
