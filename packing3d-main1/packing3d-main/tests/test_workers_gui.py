"""Headless tests of actual GUI callbacks and state transitions."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import time
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from packing import (
    BLPriority,
    Box,
    ComparisonDecoder,
    Container,
    DecoderConfig,
    GAConfig,
    RunOptions,
    parse_box_sizes_bytes,
    detect_cpu,
    read_execution_records,
    read_research_record,
    read_run_metadata_json,
    run_bl_ga,
    run_comparison_trial,
    write_comparison_report_json,
    write_result_json,
    write_run_metadata_json,
)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'demo'))
from gui_core import RunRecordState, parse_workers_request


class Var:
    def __init__(self,value=None):self.value=value;self.callbacks=[]
    def get(self):return self.value
    def set(self,value):
        self.value=value
        for callback in self.callbacks:callback()
    def trace_add(self,mode,callback):self.callbacks.append(callback)


class Widget:
    def __init__(self):self.config={};self.states=set()
    def configure(self,**kwargs):self.config.update(kwargs)
    def state(self,states):
        for s in states:
            if s.startswith('!'):self.states.discard(s[1:])
            else:self.states.add(s)
    def title(self,*_):pass
    def geometry(self,*_):pass
    def minsize(self,*_):pass
    def after(self,*_):pass
    def protocol(self,*_):pass
    def destroy(self):pass
    def delete(self,*_):pass
    def insert(self,*args,**kwargs):pass
    def get_children(self):return ()


@pytest.fixture
def gui(monkeypatch):
    loader=importlib.machinery.SourceFileLoader('packing_workers_gui',str(ROOT/'demo'/'gui.pyw'))
    spec=importlib.util.spec_from_loader(loader.name,loader);module=importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    for name in ('StringVar','DoubleVar','BooleanVar'):monkeypatch.setattr(module.tk,name,Var)
    names=('pack_button','ga_button','open_input_button','sample_button','open_json_button','open_metadata_button',
           'priority_combo','ga_decoder_combo','candidate_policy_combo','min_support_percent_entry',
           'ga_workers_mode_combo','ga_workers_entry','ga_cache_check',
           'ga_paper_defaults_button','ga_quick_button','validate_button','save_json_button','save_unity_button',
           'copy_unity_button','save_metadata_button','research_save_button','experiment_button',
           'tree','json_preview','unity_preview')
    def build(app):
        for name in names:setattr(app,name,Widget())
        app.ga_entries=[Widget() for _ in range(5)]
    monkeypatch.setattr(module.PackingDemoApp,'_build_ui',build)
    app=module.PackingDemoApp(Widget())
    app.errors=[]
    monkeypatch.setattr(app,'_show_error',lambda title,exc:app.errors.append((title,exc)))
    return module,app


@pytest.mark.parametrize('mode',['Auto','Max'])
def test_unused_manual_text(mode):
    assert parse_workers_request(mode,'not an integer').manual_workers is None
    with pytest.raises(ValueError):parse_workers_request('Manual','bad')


def test_state_and_save_retry(tmp_path):
    run=run_bl_ga(Container(2,2,2),(Box(1,1,1,1),),config=GAConfig(1,1))
    state=RunRecordState();state.complete(run)
    assert state.saved_path is None
    assert not state.save('')
    def fail(*_):raise OSError('fail')
    with pytest.raises(OSError):state.save(tmp_path/'bad.json',writer=fail)
    assert state.metadata is run.run_metadata and state.saved_path is None
    path=tmp_path/'record.json';assert state.save(path)
    assert state.saved_path==path
    with pytest.raises(OSError):state.save(tmp_path/'bad.json',writer=fail)
    assert state.saved_path==path
    state.clear();assert state.metadata is None


def test_gui_mode_controls_and_restore(gui):
    module,app=gui
    assert app.candidate_policy_var.get()=='Cartesian'
    assert app.min_support_percent_var.get()=='55'
    assert app.decoder_info_var.get()=='decoder: Cartesian v1 / 支持55% (support-union-v1)'
    assert app.ga_workers_mode_var.get()=='Auto'
    assert app.ga_workers_entry.config['state']=='disabled'
    app.ga_workers_var.set('invalid unused value')
    assert '実行前の推定値' in app.workers_info_var.get()
    app.ga_workers_mode_var.set('Manual')
    assert app.ga_workers_entry.config['state']=='normal'
    app.ga_workers_var.set('1')
    app._running=True;app._set_input_actions(False)
    assert app.ga_workers_entry.config['state']=='disabled'
    app._running=False;app._set_input_actions(True)
    assert app.ga_workers_entry.config['state']=='normal'
    app.ga_workers_mode_var.set('Max')
    app._set_input_actions(True)
    assert app.ga_workers_entry.config['state']=='disabled'
    assert app.candidate_policy_combo.config['state']=='readonly'
    assert app.min_support_percent_entry.config['state']=='normal'
    app._set_input_actions(False)
    assert app.candidate_policy_combo.config['state']=='disabled'
    assert app.min_support_percent_entry.config['state']=='disabled'


def test_gui_record_save_clear_and_widget_edit(gui,tmp_path,monkeypatch):
    module,app=gui
    decoder_config=DecoderConfig(candidate_policy='LocalNeighbors', min_support_percent=55)
    run=run_bl_ga(Container(2,2,2),(Box(1,1,1,1),),config=GAConfig(1,1),
                  decoder_config=decoder_config)
    app.record_state.complete(run)
    monkeypatch.setattr(module.filedialog,'asksaveasfilename',lambda **_: '')
    app.save_run_metadata();assert app.record_state.saved_path is None
    target=tmp_path/'saved.json'
    monkeypatch.setattr(module.filedialog,'asksaveasfilename',lambda **_:str(target))
    app.save_run_metadata();assert '保存済み' in app.record_info_var.get()
    before=app.record_state.metadata.to_dict()
    app.ga_seed_var.set('321')
    app.candidate_policy_var.set('Cartesian')
    app.min_support_percent_var.set('0')
    assert app.record_state.metadata.to_dict()==before
    assert read_run_metadata_json(target).to_dict()==before
    assert before['decoder']['candidate_policy']=='LocalNeighbors'
    assert before['decoder']['min_support_percent']==55
    app._clear_result()
    assert app.record_state.metadata is None and 'disabled' in app.save_metadata_button.states


def test_gui_manual_load_no_execution_and_cancel_atomic(gui,monkeypatch,tmp_path):
    module,app=gui
    original=ROOT/'tests'/'fixtures'/'run_metadata_v2.json'
    monkeypatch.setattr(module.filedialog,'askopenfilename',lambda **_:str(original))
    app.open_run_metadata()
    assert app.ga_workers_mode_var.get()=='Manual' and app.ga_workers_var.get()=='4'
    assert app.ga_cache_var.get() is False and app.replay_settings is not None
    assert app.ga_result is None and not app._running and app.record_state.metadata is None
    assert '未実行' in app.status_var.get()
    assert app.replay_settings.options.replay_of==read_run_metadata_json(original)['run_id']
    assert app.candidate_policy_var.get()=='Cartesian'
    assert app.min_support_percent_var.get()=='55'
    previous=(app.container,app.boxes,app.ga_workers_var.get(),app.ga_seed_var.get(),
              app.candidate_policy_var.get(),app.min_support_percent_var.get())
    bad=tmp_path/'bad.json';bad.write_text('{')
    monkeypatch.setattr(module.filedialog,'askopenfilename',lambda **_:str(bad))
    app.open_run_metadata()
    assert previous==(app.container,app.boxes,app.ga_workers_var.get(),app.ga_seed_var.get(),
                      app.candidate_policy_var.get(),app.min_support_percent_var.get())
    assert app.errors
    app.ga_seed_var.set('11')
    assert app.replay_settings is None
    assert app.problem_options.order_mode=='source'


def test_preview_vs_execution_thread_value_and_success(gui,monkeypatch):
    module,app=gui
    # A preview CPU estimate of 100 does not become the execution value.
    app.ga_population_var.set('1');app.ga_generations_var.set('1')
    app.candidate_policy_var.set('LocalNeighbors')
    app.min_support_percent_var.set('55')
    app.container=Container(2,2,2);app.boxes=(Box(1,1,1,1),)
    calls=[]
    def actual_cpu():calls.append(1);return detect_cpu(SimpleNamespace(process_cpu_count=lambda:1))
    app.problem_options=RunOptions(origin='gui',cpu_reader=actual_cpu)
    app.ga_workers_var.set('invalid unused')
    from threading import Thread
    threads=[]
    class TrackingThread(Thread):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);threads.append(self)
    monkeypatch.setattr(module.threading,'Thread',TrackingThread)
    app.start_ga()
    assert app._running and app.ga_workers_entry.config['state']=='disabled'
    assert app.candidate_policy_combo.config['state']=='disabled'
    assert app.min_support_percent_entry.config['state']=='disabled'
    for thread in threads:thread.join(timeout=10)
    app._poll_pack_queue()
    assert not app._running and not app.errors
    assert len(calls)==1 and app.ga_result.execution.workers==1
    assert app.ga_result.decoder_config == DecoderConfig(
        candidate_policy='LocalNeighbors', min_support_percent=55
    )
    assert app.ga_result.run_metadata['decoder']['candidate_policy']=='LocalNeighbors'
    assert app.ga_result.run_metadata['decoder']['min_support_percent']==55
    assert '確定' in app.workers_info_var.get() and '未保存' in app.record_info_var.get()
    assert 'disabled' not in app.save_metadata_button.states
    assert app.ga_workers_entry.config['state']=='disabled'
    assert app.candidate_policy_combo.config['state']=='readonly'
    assert app.min_support_percent_entry.config['state']=='normal'


@pytest.mark.parametrize(("decoder", "fails"), [("BL", False), ("PP", False), ("BL", True), ("PP", True)])
def test_gui_background_thread_uses_spawn_and_restores_after_success_or_failure(
    gui, monkeypatch, tmp_path, decoder: str, fails: bool
):
    import multiprocessing
    import threading
    import time

    import packing.execution as execution_module
    from spawn_helpers import (
        gui_spawn_bl_worker,
        gui_spawn_failing_bl_worker,
        gui_spawn_failing_pp_worker,
        gui_spawn_pp_worker,
    )

    module, app = gui
    marker_dir = tmp_path / "spawn-markers"
    monkeypatch.setenv("PACKING3D_GUI_SPAWN_MARKER_DIR", str(marker_dir))
    monkeypatch.setattr(execution_module, "_bl_worker", gui_spawn_bl_worker)
    if decoder == "PP":
        monkeypatch.setattr(execution_module, "_pp_worker", gui_spawn_pp_worker)
    if fails:
        failing_worker = gui_spawn_failing_bl_worker if decoder == "BL" else gui_spawn_failing_pp_worker
        target = "_bl_worker" if decoder == "BL" else "_pp_worker"
        monkeypatch.setattr(execution_module, target, failing_worker)

    runtimes = []
    original_enter = execution_module.EvaluationRuntime.__enter__

    def tracked_enter(runtime):
        entered_runtime = original_enter(runtime)
        if runtime.config.workers > 1:
            runtimes.append(runtime)
        return entered_runtime

    monkeypatch.setattr(execution_module.EvaluationRuntime, "__enter__", tracked_enter)

    entered_core = threading.Event()
    release_core = threading.Event()
    original_run_ga_core = module.run_ga_core

    def gated_run_ga_core(*args, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        entered_core.set()
        if not release_core.wait(timeout=15):
            raise TimeoutError("test did not release the GUI worker")
        return original_run_ga_core(*args, **kwargs)

    monkeypatch.setattr(module, "run_ga_core", gated_run_ga_core)
    app.container = Container(3, 2, 1)
    app.boxes = (Box(1, 2, 1, 1), Box(2, 1, 1, 1))
    app.ga_decoder_var.set(decoder)
    app.ga_population_var.set("2")
    app.ga_generations_var.set("2")
    app.ga_crossover_var.set("0.8")
    app.ga_mutation_var.set("0.1")
    app.ga_seed_var.set("23")
    app.ga_workers_mode_var.set("Manual")
    app.ga_workers_var.set("2")
    app.ga_cache_var.set(False)

    app.start_ga()

    assert entered_core.wait(timeout=10)
    assert app._running
    assert app.ga_workers_entry.config["state"] == "disabled"
    assert app.candidate_policy_combo.config["state"] == "disabled"
    assert app.min_support_percent_entry.config["state"] == "disabled"
    release_core.set()

    deadline = time.monotonic() + 60
    while app._running and time.monotonic() < deadline:
        app._poll_pack_queue()
        time.sleep(0.01)
    app._poll_pack_queue()

    assert not app._running
    assert runtimes and all(runtime._executor is None for runtime in runtimes)
    marker_files = tuple(marker_dir.glob("*.marker"))
    assert marker_files
    worker_pids = {int(path.stem) for path in marker_files}
    assert not any(child.pid in worker_pids and child.is_alive() for child in multiprocessing.active_children())
    assert app.ga_workers_entry.config["state"] == "normal"
    assert app.candidate_policy_combo.config["state"] == "readonly"
    assert app.min_support_percent_entry.config["state"] == "normal"

    if fails:
        assert app.errors and "GUI" in str(app.errors[-1][1])
        assert app.ga_result is None
        assert app.record_state.metadata is None
        assert "失敗" in app.status_var.get()
    else:
        assert not app.errors
        assert app.ga_result is not None
        assert app.ga_result.execution.workers == 2
        assert app.ga_result.run_metadata["execution"]["start_method"] == "spawn"
        assert app.record_state.metadata is app.ga_result.run_metadata
        assert app.ga_progress_percent_var.get() == 100.0


def test_preflight_failure_does_not_reuse_record(gui,monkeypatch):
    module,app=gui
    app.container=Container(2,2,2);app.boxes=(Box(1,1,1,1),)
    app.ga_population_var.set('1');app.ga_generations_var.set('1')
    app.ga_workers_mode_var.set('Manual');app.ga_workers_var.set('62')
    old=run_bl_ga(app.container,app.boxes,config=GAConfig(1,1))
    app.record_state.complete(old)
    import packing.workers as workers
    monkeypatch.setattr(workers.platform,'system',lambda:'Windows')
    from threading import Thread
    threads=[]
    class TrackingThread(Thread):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);threads.append(self)
    monkeypatch.setattr(module.threading,'Thread',TrackingThread)
    app.start_ga()
    for thread in threads:thread.join(timeout=10)
    app._poll_pack_queue()
    assert app.errors and app.record_state.metadata is None
    assert app.ga_workers_entry.config['state']=='normal' and not app._running


def test_gui_bl_record_snapshot_keeps_replay_source_order(gui, monkeypatch):
    module, app = gui
    source_boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 3))
    app.container = Container(3, 2, 3)
    app.boxes = (source_boxes[1], source_boxes[0])
    app.problem_options = RunOptions(
        origin="gui", source_boxes=source_boxes, order_mode="depth_desc_stable",
    )

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=ImmediateThread))
    app.start_pack()
    app._poll_pack_queue()

    assert app.research_capture is not None
    problem = app.research_capture.snapshot["problem"]
    assert problem["source_box_order"] == [1, 2]
    assert problem["effective_box_order"] == [2, 1]


def test_root_close_requests_experiment_stop_without_destroying_root(gui):
    _module, app = gui
    calls = []
    app.root.destroy = lambda: calls.append("destroy")

    class ActiveDialog:
        worker = object()

        def request_close(self):
            calls.append("stop")

    app._research_experiment_dialog = ActiveDialog()
    app.close_window()

    assert calls == ["stop"]
    assert app._close_when_experiment_done


def test_gui_invalid_decoder_setting_does_not_start_or_replace_completed_run(gui):
    _,app=gui
    app.container=Container(2,2,2)
    app.boxes=(Box(1,1,1,1),)
    old=run_bl_ga(app.container,app.boxes,config=GAConfig(1,1))
    app.ga_result=old
    app.record_state.complete(old)
    app.save_metadata_button.state(['!disabled'])
    app.min_support_percent_var.set('101')

    app.start_ga()

    assert app.errors and not app._running
    assert app.ga_result is old
    assert app.record_state.metadata is old.run_metadata
    assert 'disabled' not in app.save_metadata_button.states


def test_gui_multiple_trial_cancel_and_input_failure_no_partial_update(gui,monkeypatch,tmp_path):
    module,app=gui
    app.container=Container(2,2,2);app.boxes=(Box(1,1,1,1),)
    report=ROOT/'tests'/'fixtures'/'comparison_v4.json'
    monkeypatch.setattr(module.filedialog,'askopenfilename',lambda **_:str(report))
    monkeypatch.setattr(app,'_select_record',lambda _:None)
    before=(app.container,app.boxes,app.ga_seed_var.get())
    app.open_run_metadata()
    assert before==(app.container,app.boxes,app.ga_seed_var.get())
    invalid=tmp_path/'bad.txt';invalid.write_text('bad')
    app._load_box_sizes(invalid,str(invalid))
    assert before==(app.container,app.boxes,app.ga_seed_var.get())


def test_gui_replay_edit_preserves_depth_order(gui,monkeypatch,tmp_path):
    module,app=gui
    from packing import BoxOrderMode, ComparisonDecoder, run_comparison_trial, write_run_metadata_json
    trial=run_comparison_trial(Container(3,3,3),(Box(4,1,1,1),Box(2,1,1,2)),
                              decoder=ComparisonDecoder.BL,order_mode=BoxOrderMode.DEPTH_DESC_STABLE,
                              config=GAConfig(1,1))
    path=tmp_path/'metadata.json';write_run_metadata_json(path,trial.run_metadata)
    monkeypatch.setattr(module.filedialog,'askopenfilename',lambda **_:str(path))
    app.open_run_metadata()
    assert app.replay_settings is not None
    app.ga_seed_var.set('4')
    assert app.replay_settings is None
    assert app.problem_options.order_mode=='depth_desc_stable'
    assert [b.id for b in app.problem_options.source_boxes]==[4,2]
    assert [b.id for b in app.boxes]==[2,4]


def test_gui_v2_replay_execution_keeps_restored_decoder_config(gui,monkeypatch,tmp_path):
    module,app=gui
    decoder_config=DecoderConfig(candidate_policy='LocalNeighbors', min_support_percent=55)
    old=run_bl_ga(Container(2,2,2),(Box(1,1,1,1),),config=GAConfig(1,1),
                  decoder_config=decoder_config)
    path=tmp_path/'metadata-v2.json'
    write_run_metadata_json(path,old.run_metadata)
    monkeypatch.setattr(module.filedialog,'askopenfilename',lambda **_:str(path))
    app.open_run_metadata()
    assert app.replay_settings.decoder_config==decoder_config
    assert app.candidate_policy_var.get()=='LocalNeighbors'
    assert app.min_support_percent_var.get()=='55'

    from threading import Thread
    threads=[]
    class TrackingThread(Thread):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            threads.append(self)
    monkeypatch.setattr(module.threading,'Thread',TrackingThread)
    app.start_ga()
    for thread in threads:thread.join(timeout=10)
    app._poll_pack_queue()
    assert not app.errors
    assert app.ga_result.decoder_config==decoder_config
    assert app.ga_result.run_metadata['decoder']['candidate_policy']=='LocalNeighbors'
    assert app.ga_result.run_metadata['decoder']['min_support_percent']==55


def test_gui_selects_v4_trial_and_restores_all_decoder_settings(gui,monkeypatch,tmp_path):
    module,app=gui
    container=Container(3,2,2)
    boxes=(Box(5,1,1,1),Box(2,1,1,2),Box(9,1,2,1))
    config=GAConfig(1,1)
    settings=(
        DecoderConfig(min_support_percent=0),
        DecoderConfig(candidate_policy='LocalNeighbors',min_support_percent=55),
    )
    trials=[]
    metadata_values=[]
    for decoder,decoder_config in zip((ComparisonDecoder.BL,ComparisonDecoder.PP),settings):
        trial=run_comparison_trial(
            container,boxes,decoder=decoder,priority=BLPriority.BACK_FIRST,
            config=config,seed=23,decoder_config=decoder_config,
        )
        metadata=trial.run_metadata.to_dict()
        metadata_values.append(metadata)
        trials.append({
            'run_metadata':metadata,
            'best':{
                'generation_index':trial.best_generation_index,
                'population_index':trial.best_population_index,
                'genes':list(trial.best_genes),
                'fitness':trial.best_fitness,
                'placed_count':trial.placed_count,
                'failed_count':trial.failed_count,
                'placed_volume':trial.placed_volume,
                'density':trial.density,
            },
        })
    report=tmp_path/'comparison-v4.json'
    write_comparison_report_json(report,{
        'format':'packing3d-comparison-report',
        'version':4,
        'implementation_version':metadata_values[0]['environment']['implementation']['version'],
        'trials':trials,
    })
    monkeypatch.setattr(module.filedialog,'askopenfilename',lambda **_:str(report))
    monkeypatch.setattr(app,'_select_record',lambda records:records[1])

    app.open_run_metadata()

    assert app.replay_settings.decoder_config==settings[1]
    assert app.candidate_policy_var.get()=='LocalNeighbors'
    assert app.min_support_percent_var.get()=='55'
    assert app.ga_decoder_var.get()=='PP'
    assert app.ga_result is None and app.record_state.metadata is None and not app._running
    assert '未実行' in app.status_var.get()
    assert 'LocalNeighbors v1 / 支持55% (support-union-v1)' in module.PackingDemoApp._record_selection_label(
        read_execution_records(report)[1]
    )

    app.min_support_percent_var.set('54')
    assert app.replay_settings is None
    assert '保存した条件での再実行ではなくなりました' in app.status_var.get()




def test_gui_bl_execution_uses_selected_decoder_settings(gui,monkeypatch):
    module,app=gui
    app.container=Container(3,2,2)
    app.boxes=(Box(5,1,1,1),Box(2,1,1,2),Box(9,1,2,1))
    app.candidate_policy_var.set('LocalNeighbors')
    app.min_support_percent_var.set('55')
    from threading import Thread
    threads=[]
    class TrackingThread(Thread):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            threads.append(self)
    monkeypatch.setattr(module.threading,'Thread',TrackingThread)

    app.start_pack()
    assert app._running
    assert app.candidate_policy_combo.config['state']=='disabled'
    assert app.min_support_percent_entry.config['state']=='disabled'
    for thread in threads:thread.join(timeout=10)
    app._poll_pack_queue()

    assert not app.errors
    assert app.result_decoder_config==DecoderConfig(
        candidate_policy='LocalNeighbors',min_support_percent=55
    )
    assert 'LocalNeighbors v1' in app.ga_info_var.get()
    assert app.candidate_policy_combo.config['state']=='readonly'
    assert app.min_support_percent_entry.config['state']=='normal'


def test_gui_support_validation_uses_result_snapshot(gui,monkeypatch):
    module,app=gui
    config=DecoderConfig(candidate_policy='LocalNeighbors',min_support_percent=100)
    run=run_bl_ga(
        Container(2,2,2),(Box(1,1,1,1),),config=GAConfig(1,1),decoder_config=config
    )
    app.container=run.container
    app.result_expected_boxes=run.boxes
    app._show_result(
        run.best.packing_result,
        origin='ga',
        expected_boxes=run.boxes,
        decoder_config=run.decoder_config,
        run_info='GA result',
    )
    observed=[]
    real_validate_support=module.validate_support
    def capture(result,*,settings):
        observed.append(settings)
        return real_validate_support(result,settings=settings)
    monkeypatch.setattr(module,'validate_support',capture)
    monkeypatch.setattr(module.messagebox,'showinfo',lambda *args,**kwargs:None)

    app.validate_current_result()

    assert observed==[config]
    assert app.validation_var.get()=='検証OK / 支持100%確認'


def test_gui_result_json_marks_support_unchecked_and_clears_run_record(gui,monkeypatch,tmp_path):
    module,app=gui
    run=run_bl_ga(Container(2,2,2),(Box(1,1,1,1),),config=GAConfig(1,1))
    app.record_state.complete(run)
    app.ga_result=run
    app.result_decoder_config=run.decoder_config
    result_path=tmp_path/'result.json'
    write_result_json(result_path,run.best.packing_result)
    monkeypatch.setattr(module.filedialog,'askopenfilename',lambda **_:str(result_path))
    notices=[]
    monkeypatch.setattr(module.messagebox,'showinfo',lambda *args,**kwargs:notices.append(args))

    app.open_result_json()
    assert app.result_decoder_config is None
    assert app.record_state.metadata is None
    assert 'disabled' in app.save_metadata_button.states
    assert '支持未検査' in app.validation_var.get()
    assert '支持は未検査' in app.status_var.get()

    app.validate_current_result()
    assert app.validation_var.get()=='配置範囲・重なり: 問題なし / 支持条件: 未確認'
    assert notices and '支持条件は実行時の設定が不明なため確認できません' in notices[0][1]
    assert app.research_capture is None
    assert 'disabled' in app.research_save_button.states


@pytest.mark.parametrize(('mode','full'),[('BL',False),('GA',False),('GA',True)])
def test_gui_generated_run_has_capture_and_can_save_research_record(gui,monkeypatch,tmp_path,mode,full):
    module,app=gui
    raw=b'{3,3,3}{1,1,1}{1,1,1}'
    app.container,app.boxes=parse_box_sizes_bytes(raw,source='synthetic.txt')
    app.problem_options=RunOptions(origin='gui',source='synthetic.txt',fixture_sha256=hashlib.sha256(raw).hexdigest())
    app.source_bytes=raw
    app.pack_button.state(['!disabled']);app.ga_button.state(['!disabled'])
    if mode=='BL':
        app.start_pack()
    else:
        app.ga_population_var.set('2');app.ga_generations_var.set('1')
        app.ga_workers_mode_var.set('Manual');app.ga_workers_var.set('1')
        app.start_ga()
    deadline=time.monotonic()+10
    while app._running and time.monotonic()<deadline:
        time.sleep(.01);app._poll_pack_queue()
    assert not app._running
    assert app.result_origin in ('bl','ga')
    assert app.research_capture is not None
    assert 'disabled' not in app.research_save_button.states
    app.candidate_policy_var.set('LocalNeighbors')
    app.min_support_percent_var.set('0')
    app.source_bytes=b'changed after run'

    monkeypatch.setattr(module.filedialog,'askdirectory',lambda **_:str(tmp_path))
    monkeypatch.setattr(module.messagebox,'askyesnocancel',lambda *_,**__:full)
    app.save_research_record()
    records=list((tmp_path/'records').glob('*/record.json'))
    assert len(records)==1
    saved=read_research_record(records[0].parent)
    assert saved['status']=='completed'
    assert saved['record_type']==('BL' if mode=='BL' else 'BL-GA')
    assert saved['save_level']==('full' if full else 'summary')
    assert saved['snapshot']['problem']['fixture_sha256']==hashlib.sha256(raw).hexdigest()
    assert saved['snapshot']['decoder']['candidate_policy']=='Cartesian'
    assert (records[0].parent/'source.txt').read_bytes()==raw
    if mode=='GA' and not full:
        original_id=saved['record_id']
        original_run_id=saved['ga_run_id']
        original_result=(records[0].parent/'best.result.json').read_bytes()
        app.save_research_record()
        repeated=sorted((tmp_path/'records').glob('*/record.json'))
        assert len(repeated)==2
        repeated_records=[read_research_record(path.parent) for path in repeated]
        assert len({record['record_id'] for record in repeated_records})==2
        assert {record['ga_run_id'] for record in repeated_records}=={original_run_id}
        assert any(record['record_id']==original_id for record in repeated_records)
        assert (records[0].parent/'best.result.json').read_bytes()==original_result


def test_experiment_lock_disables_main_gui_actions(gui):
    _module,app=gui
    app.container=Container(2,2,2);app.boxes=(Box(1,1,1,1),)
    app._research_experiment_active=True
    app._set_input_actions(True)
    assert 'disabled' in app.experiment_button.states
    assert app.open_input_button.states=={'disabled'}
    assert app.ga_workers_entry.config['state']=='disabled'
