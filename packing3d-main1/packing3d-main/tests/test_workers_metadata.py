from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from packing import (BLPriority, Box, Container, DecoderConfig, GAConfig, GAExecutionConfig, InputFormatError,
    RunMetadata, RunOptions, WorkersRequest, detect_cpu, resolve_workers, preflight_workers,
    run_bl_ga, run_pp_ga, prepare_manual_replay, run_manual_replay, read_run_metadata_json,
    write_run_metadata_json, validate_run_metadata, read_comparison_report_json,
    write_comparison_report_json, validate_comparison_report, read_execution_records)
from packing.io_run_metadata import render_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'


def cpu(a=4):
    return detect_cpu(SimpleNamespace(process_cpu_count=lambda: a))


@pytest.fixture
def metadata():
    return read_run_metadata_json(FIXTURES / 'run_metadata_v2.json').to_dict()


@pytest.fixture
def metadata_v2():
    return read_run_metadata_json(FIXTURES / 'run_metadata_v2.json').to_dict()


@pytest.mark.parametrize('a,p,l,auto,max_workers', [
    (1,100,None,1,1),(2,100,None,1,2),(3,100,None,2,3),(4,100,None,3,4),
    (16,100,None,8,16),(100,1,None,1,1),(100,100,61,8,61),(100,61,61,8,61),
    (16,8,8,8,8),(12,8,None,8,8),(8,100,None,6,8)])
def test_resolution_table(a,p,l,auto,max_workers):
    assert resolve_workers(WorkersRequest('auto',None),a,p,l)['resolved_workers'] == auto
    assert resolve_workers(WorkersRequest('max',None),a,p,l)['resolved_workers'] == max_workers


def test_tied_limits_and_unchanged_boundary():
    assert resolve_workers(WorkersRequest('auto',None),16,8,8)['limit_reasons'] == ['auto_cap','population_size','executor_limit']
    assert resolve_workers(WorkersRequest('max',None),100,61,61)['limit_reasons'] == ['population_size','executor_limit']
    assert resolve_workers(WorkersRequest('auto',None),11,8)['limit_reasons'] == []
    w=resolve_workers(WorkersRequest('manual',16),1,4,61)
    assert w['resolved_workers']==16 and w['limit_reasons']==[]
    assert w['warning_reasons']==['manual_exceeds_cpu_estimate','manual_exceeds_population']
    preflight_workers(61)
    with pytest.raises(ValueError): preflight_workers(62)


@pytest.mark.parametrize('value',[True,False,1.0,'2',None,0,-1])
def test_manual_invalid(value):
    with pytest.raises((TypeError,ValueError)):
        WorkersRequest('manual',value)


@pytest.mark.parametrize('mode',[None,1,'AUTO','invalid'])
def test_invalid_mode(mode):
    with pytest.raises((TypeError,ValueError)): WorkersRequest(mode,None)


@pytest.mark.parametrize('invalid,status,value',[ (None,'none',None),(0,'invalid',0),(-3,'invalid',-3),
    (True,'invalid',None),(1.5,'invalid',None),('4','invalid',None),(float('nan'),'invalid',None)])
def test_cpu_invalid_then_system_count(invalid,status,value):
    d=detect_cpu(SimpleNamespace(process_cpu_count=lambda:invalid,cpu_count=lambda: 2))
    assert d['attempts'][0]['status']==status
    assert d['attempts'][0]['value']==value
    assert d['selected_method']=='os.cpu_count' and d['available_logical_estimate']==2
    assert d['attempts'][1]['status']=='unavailable'
    if status=='invalid' and value is None:
        assert d['attempts'][0]['detail']


@pytest.mark.parametrize('exc,status',[(OSError('problem'),'os_error'),(NotImplementedError('missing'),'not_implemented')])
def test_cpu_exception_and_fallback(exc,status):
    def fail(): raise exc
    d=detect_cpu(SimpleNamespace(process_cpu_count=fail))
    assert d['attempts'][0]['status']==status
    assert d['selected_method']=='fallback' and d['available_logical_estimate']==1
    assert d['fallback_reasons']==['cpu_fallback']


def test_cpu_stop_unexpected_and_diagnostic_bound():
    def bug(): raise RuntimeError('unexpected')
    d=detect_cpu(SimpleNamespace(process_cpu_count=lambda:3,cpu_count=bug,sched_getaffinity=bug))
    assert [a['status'] for a in d['attempts']]==['success','not_called','not_called','not_called']
    with pytest.raises(RuntimeError): detect_cpu(SimpleNamespace(process_cpu_count=bug))
    d=detect_cpu(SimpleNamespace(process_cpu_count=lambda:'x'*5000))
    assert len(d['attempts'][0]['detail'])==2048
    d=detect_cpu(SimpleNamespace(sched_getaffinity=lambda _:None,cpu_count=lambda:2))
    assert d['attempts'][1]['status']=='unavailable' and d['selected_method']=='os.cpu_count'


def test_metadata_v2_fixture_roundtrip_preserves_decoder_values(metadata_v2,tmp_path):
    assert metadata_v2['version']==2
    assert tuple(metadata_v2['decoder']) == (
        'name','priority','reference_version','fitness_version','pp_initialization',
        'candidate_policy','candidate_policy_version','min_support_percent','support_model_version')
    assert metadata_v2['decoder']['min_support_percent']==55
    assert 'constraints' not in metadata_v2['decoder']
    path=tmp_path/'metadata-v2.json'
    write_run_metadata_json(path,metadata_v2)
    saved=read_run_metadata_json(path).to_dict()
    assert saved==metadata_v2
    assert tuple(saved)==(
        'format','version','run_id','status','origin','replay_of','problem','decoder',
        'ga','execution','environment','statistics')
    assert tuple(saved['decoder'])==(
        'name','priority','reference_version','fitness_version','pp_initialization',
        'candidate_policy','candidate_policy_version','min_support_percent','support_model_version')
    first=path.read_bytes()
    assert first.endswith(b'\n') and b'\r' not in first
    write_run_metadata_json(path,saved)
    assert path.read_bytes()==first
    path.write_bytes(b'\xef\xbb\xbf'+first)
    assert read_run_metadata_json(path).to_dict()==saved

    metadata_v2['decoder']['min_support_percent']=0
    write_run_metadata_json(path,metadata_v2)
    assert read_run_metadata_json(path)['decoder']['min_support_percent']==0


@pytest.mark.parametrize('policy',['Cartesian','LocalNeighbors'])
@pytest.mark.parametrize('percent',[0,100])
def test_metadata_v2_accepts_canonical_policy_and_support_boundaries(metadata_v2,policy,percent):
    metadata_v2['decoder']['candidate_policy']=policy
    metadata_v2['decoder']['min_support_percent']=percent
    assert validate_run_metadata(metadata_v2) is None


def test_metadata_v2_rejects_integer_subclasses(metadata_v2):
    class IntSubclass(int):
        pass

    metadata_v2['decoder']['candidate_policy_version']=IntSubclass(1)
    with pytest.raises(TypeError):
        validate_run_metadata(metadata_v2)
    metadata_v2['decoder']['candidate_policy_version']=1
    metadata_v2['decoder']['min_support_percent']=IntSubclass(55)
    with pytest.raises(TypeError):
        validate_run_metadata(metadata_v2)


@pytest.mark.parametrize('path,value',[
    (('version',),3),
    (('decoder','name'),'bl'),
    (('decoder','name'),1),
    (('decoder','priority'),'bottomfirst'),
    (('decoder','priority'),None),
    (('decoder','reference_version'),'PP-ref-v1'),
    (('decoder','reference_version'),'BL-ref-v2'),
    (('decoder','fitness_version'),'fitness-v2'),
    (('decoder','pp_initialization'),None),
    (('decoder','candidate_policy'),None),
    (('decoder','candidate_policy'),'unknown'),
    (('decoder','candidate_policy_version'),True),
    (('decoder','candidate_policy_version'),1.0),
    (('decoder','candidate_policy_version'),2),
    (('decoder','min_support_percent'),True),
    (('decoder','min_support_percent'),1.0),
    (('decoder','min_support_percent'),-1),
    (('decoder','min_support_percent'),101),
    (('decoder','min_support_percent'),'55'),
    (('decoder','support_model_version'),None),
    (('decoder','support_model_version'),'support-v2'),
])
def test_metadata_v2_rejects_invalid_version_and_decoder_values(metadata_v2,path,value):
    target=metadata_v2
    for key in path[:-1]: target=target[key]
    target[path[-1]]=value
    with pytest.raises((TypeError,ValueError)):
        validate_run_metadata(metadata_v2)


@pytest.mark.parametrize('key',[
    'name','priority','reference_version','fitness_version','pp_initialization',
    'candidate_policy','candidate_policy_version','min_support_percent','support_model_version',
])
def test_metadata_v2_rejects_missing_decoder_key(metadata_v2,key):
    del metadata_v2['decoder'][key]
    with pytest.raises(ValueError,match='missing or unknown keys'):
        validate_run_metadata(metadata_v2)


def test_metadata_v2_rejects_legacy_constraints_key(metadata_v2):
    metadata_v2['decoder']['constraints']=None
    with pytest.raises(ValueError,match='missing or unknown keys'):
        validate_run_metadata(metadata_v2)


def test_metadata_v2_reader_and_writer_reject_schema_errors(metadata_v2,tmp_path):
    metadata_v2['decoder']['candidate_policy_version']=True
    path=tmp_path/'invalid-v2.json'
    path.write_text(render_json(metadata_v2))
    with pytest.raises(InputFormatError,match='invalid-v2.json'):
        read_run_metadata_json(path)

    path.write_text('keep')
    with pytest.raises(TypeError):
        write_run_metadata_json(path,metadata_v2)
    assert path.read_text()=='keep'


def test_comparison_v4_fixture_roundtrip_preserves_metadata_v2(tmp_path):
    report=read_comparison_report_json(FIXTURES/'comparison_v4.json')
    assert report['version']==4
    assert report['trials'][0]['run_metadata']['version']==2
    output=tmp_path/'comparison-v4.json'
    write_comparison_report_json(output,report)
    assert read_comparison_report_json(output)==report
    serialized=json.loads(output.read_text(encoding='utf-8'))
    assert tuple(serialized)==('format','version','implementation_version','trials')
    assert tuple(serialized['trials'][0])==('run_metadata','best')
    assert tuple(serialized['trials'][0]['best'])==(
        'generation_index','population_index','genes','fitness','placed_count',
        'failed_count','placed_volume','density')
    assert tuple(serialized['trials'][0]['run_metadata']['decoder'])==(
        'name','priority','reference_version','fitness_version','pp_initialization',
        'candidate_policy','candidate_policy_version','min_support_percent','support_model_version')
    first=output.read_bytes()
    write_comparison_report_json(output,report)
    assert output.read_bytes()==first


def test_comparison_v4_reader_and_writer_reject_schema_errors(tmp_path):
    report=read_comparison_report_json(FIXTURES/'comparison_v4.json')
    report['trials'][0]['run_metadata']['decoder']['support_model_version']='unknown'
    path=tmp_path/'invalid-v4.json'
    path.write_text(render_json(report))
    with pytest.raises(InputFormatError,match='invalid-v4.json'):
        read_comparison_report_json(path)

    path.write_text('keep')
    with pytest.raises(ValueError):
        write_comparison_report_json(path,report)
    assert path.read_text()=='keep'


# All nested objects must reject unknown/missing keys, including nullable keys.
@pytest.mark.parametrize('path',[(),('problem',),('decoder',),('ga',),('execution',),('execution','workers'),
    ('environment',),('environment','os'),('environment','python'),('environment','implementation'),
    ('environment','executor'),('environment','cpu'),('environment','cpu','detection'),
    ('environment','cpu','detection','attempts',0),('statistics',)])
@pytest.mark.parametrize('operation',['extra','missing'])
def test_all_object_keys(metadata,path,operation):
    target=metadata
    for key in path: target=target[key]
    if operation=='extra': target['unknown']=None
    else: del target[next(iter(target))]
    with pytest.raises((TypeError,ValueError)): validate_run_metadata(metadata)


@pytest.mark.parametrize('path,value',[
    (('format',),'packing-result'),(('version',),1),(('version',),True),
    (('run_id',),'abc'),(('status',),'failed'),(('origin',),'other'),
    (('ga','seed'),True),(('ga','seed'),1.0),(('ga','seed'),-1),
    (('ga','crossover_rate'),float('nan')),(('ga','mutation_rate'),float('inf')),
    (('problem','container'),[1,1,True]),(('problem','source_box_order'),[]),
    (('problem','effective_box_order'),[]),(('problem','fixture_sha256'),'A'*64),
    (('decoder','name'),'bl'),(('decoder','reference_version'),'BL-ref-v1'),
    (('decoder','pp_initialization'),None),(('decoder','candidate_policy'),'unknown'),
    (('decoder','constraints'),{}),(('execution','cache'),1),
    (('execution','workers','requested_mode'),'auto'),(('execution','workers','manual_workers'),2),
    (('execution','workers','resolver_version'),'new'),(('execution','workers','auto_cap'),9),
    (('execution','workers','auto_ratio'),0.8),(('execution','workers','limit_reasons'),['auto_cap']),
    (('execution','workers','warning_reasons'),['unknown']),
    (('execution','start_method'),None),(('statistics','requested'),0),
    (('statistics','cache_hits'),1),(('statistics','elapsed_scope'),'unknown'),
    (('environment','executor','worker_limit'),None),(('environment','cpu','label_source'),'platform.processor'),
    (('environment','cpu','detection','selected_method'),'fallback'),
    (('environment','cpu','detection','available_logical_estimate'),12345),
    (('environment','cpu','detection','attempts',0,'status'),'not_called'),
    (('environment','cpu','detection','attempts',1,'status'),'success'),
    (('environment','cpu','detection','attempts',3,'detail'),'non-null'),
])
def test_inconsistent_metadata(metadata,path,value,tmp_path):
    target=metadata
    for key in path[:-1]: target=target[key]
    target[path[-1]]=value
    with pytest.raises((TypeError,ValueError)): validate_run_metadata(metadata)
    dest=tmp_path/'existing.json';dest.write_text('keep')
    with pytest.raises((TypeError,ValueError)):write_run_metadata_json(dest,metadata)
    assert dest.read_text()=='keep'


@pytest.mark.parametrize('fixture_path',[
    FIXTURES/'run_metadata_v2.json',
])
def test_atomic_os_failure(fixture_path,tmp_path,monkeypatch):
    import packing.io_run_metadata as io
    metadata=read_run_metadata_json(fixture_path).to_dict()
    dest=tmp_path/'existing.json';dest.write_text('keep')
    def fail(*_): raise OSError('replace failed')
    monkeypatch.setattr(io.os,'replace',fail)
    with pytest.raises(OSError): write_run_metadata_json(dest,metadata)
    assert dest.read_text()=='keep'
    assert list(tmp_path.iterdir())==[dest]


@pytest.mark.parametrize('fixture_path',[
    FIXTURES/'run_metadata_v2.json',
])
@pytest.mark.parametrize('kind',['duplicate','exponent','bool','nan','infinity','utf8','malformed'])
def test_reader_errors(fixture_path,tmp_path,kind):
    metadata=read_run_metadata_json(fixture_path).to_dict()
    text=render_json(metadata)
    version_token=f'"version": {metadata["version"]},'
    if kind=='duplicate': text=text.replace(version_token,version_token+f' "version": {metadata["version"]},')
    if kind=='exponent': text=text.replace(version_token,f'"version": {metadata["version"]}.0,')
    if kind=='bool': text=text.replace(version_token,'"version": true,')
    if kind=='nan': text=text.replace(version_token,'"version": NaN,')
    if kind=='infinity': text=text.replace(version_token,'"version": Infinity,')
    if kind=='malformed': text='{'
    path=tmp_path/'bad.json'
    path.write_bytes(b'\xff' if kind=='utf8' else text.encode())
    with pytest.raises(InputFormatError,match='bad.json'):read_run_metadata_json(path)


@pytest.mark.parametrize('fixture_path',[
    FIXTURES/'run_metadata_v2.json',
])
def test_1000_digit_roundtrip_low_guard(fixture_path,tmp_path):
    metadata=read_run_metadata_json(fixture_path).to_dict()
    original=sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        metadata['ga']['seed']=10**999
        path=tmp_path/'large.json'
        write_run_metadata_json(path,metadata)
        assert read_run_metadata_json(path)['ga']['seed']==10**999
        assert sys.get_int_max_str_digits()==640
        metadata['ga']['seed']=2**53+1
        write_run_metadata_json(path,metadata)
        assert read_run_metadata_json(path)['ga']['seed']==2**53+1
        metadata['ga']['seed']=10**1000
        with pytest.raises(ValueError): write_run_metadata_json(path,metadata)
        path.write_text(render_json({**metadata,'ga':{**metadata['ga'],'seed':0}}).replace('"seed": 0','"seed": '+'1'*1001))
        with pytest.raises(InputFormatError):read_run_metadata_json(path)
    finally:sys.set_int_max_str_digits(original)


@pytest.mark.parametrize('runner',[run_bl_ga,run_pp_ga])
@pytest.mark.parametrize('priority',list(BLPriority))
@pytest.mark.parametrize('cache',[False,True])
def test_full_ga_semantics_auto_max_manual(runner,priority,cache):
    c=Container(3,2,2); boxes=(Box(3,1,1,1),Box(1,2,1,1),Box(8,1,2,1))
    config=GAConfig(4,3,0.8,0.2)
    decoder_config=DecoderConfig(min_support_percent=55)
    baseline=runner(c,boxes,priority=priority,config=config,seed=91,execution=GAExecutionConfig(1,False),
                    decoder_config=decoder_config)
    for mode in ('manual','auto','max'):
        calls=[]
        def reader():calls.append(1);return cpu(3)
        options=RunOptions(request=WorkersRequest(mode,2 if mode=='manual' else None),cpu_reader=reader)
        run=runner(c,boxes,priority=priority,config=config,seed=91,execution=GAExecutionConfig(1,cache),
                   run_options=options,decoder_config=decoder_config)
        assert run==baseline # Includes all generations, individuals, PP input states and best.
        assert len(calls)==1
        assert run.run_metadata['execution']['workers']['resolved_workers']==run.execution.workers
        assert run.run_metadata['statistics']['requested']==12
        assert run.run_metadata['statistics']['computed']==run.execution_stats.computed
        assert run.run_metadata['decoder']['min_support_percent']==55


@pytest.mark.parametrize('runner',[run_bl_ga,run_pp_ga])
@pytest.mark.parametrize('cache',[False,True])
def test_empty_problem_and_fallback(runner,cache):
    run=runner(Container(1,1,1),(),config=GAConfig(2,2),execution=GAExecutionConfig(1,cache),
               run_options=RunOptions(request=WorkersRequest('auto',None),cpu_reader=lambda:detect_cpu(SimpleNamespace())))
    assert run.run_metadata['statistics']['requested']==4
    assert run.run_metadata['environment']['cpu']['detection']['selected_method']=='fallback'
    assert run.run_metadata['execution']['start_method'] is None


@pytest.mark.parametrize('runner',[run_bl_ga,run_pp_ga])
def test_omitted_decoder_config_uses_cartesian_55_support_in_new_metadata(runner):
    run=runner(Container(2,1,1),(Box(1,1,1,1),),config=GAConfig(1,1))
    decoder=run.run_metadata['decoder']

    assert run.decoder_config==DecoderConfig()
    assert run.run_metadata['version']==2
    assert decoder['reference_version']==f"{run.decoder_name}-ref-v2"
    assert decoder['candidate_policy']=='Cartesian'
    assert decoder['candidate_policy_version']==1
    assert decoder['min_support_percent']==55
    assert decoder['support_model_version']=='support-union-v1'
    assert 'constraints' not in decoder


def test_snapshot_detached_and_cpu_once(metadata):
    snapshot=RunMetadata(metadata)
    metadata['ga']['seed']=123
    changed=snapshot['ga'];changed['seed']=555
    assert snapshot['ga']['seed']!=123 and snapshot['ga']['seed']!=555
    with pytest.raises((AttributeError,TypeError)):snapshot._text='bad'


def test_replay_preserves_order_and_sets_new_id(tmp_path):
    from packing import BoxOrderMode, ComparisonDecoder, run_comparison_trial
    c=Container(3,3,3);boxes=(Box(5,1,1,1),Box(2,1,1,2),Box(9,1,1,2))
    trial=run_comparison_trial(c,boxes,decoder=ComparisonDecoder.PP,order_mode=BoxOrderMode.DEPTH_DESC_STABLE,config=GAConfig(2,2))
    old=trial.run_metadata
    settings=prepare_manual_replay(old)
    assert [b.id for b in settings.boxes]==[2,9,5]
    assert settings.decoder_config==DecoderConfig(min_support_percent=55)
    new=run_manual_replay(old)
    assert new.run_metadata['replay_of']==old['run_id']
    assert new.run_metadata['run_id']!=old['run_id']
    assert new.run_metadata['problem']==old['problem']
    assert new.run_metadata['execution']['workers']['requested_mode']=='manual'
    assert new.run_metadata['decoder']['min_support_percent']==old['decoder']['min_support_percent']


def test_recorded_os_does_not_bypass_windows_preflight(metadata):
    metadata['environment']['os']['system']='Linux'
    metadata['environment']['executor']['worker_limit']=None
    metadata['environment']['executor']['limit_source']='no_known_limit'
    metadata['execution']['workers']=resolve_workers(WorkersRequest('manual',62),
        metadata['environment']['cpu']['detection']['available_logical_estimate'],metadata['ga']['population_size'])
    metadata['execution']['start_method']='spawn'
    validate_run_metadata(metadata)
    with pytest.raises(ValueError, match='61'):
        prepare_manual_replay(metadata)


@pytest.mark.parametrize('mutation',['root_version','duplicate','implementation','best_index','genes','density','counts','extra'])
def test_comparison_rejects(mutation):
    source=FIXTURES/'comparison_v4.json'
    report=read_comparison_report_json(source)
    t=report['trials'][0]
    if mutation=='root_version':report['version']=99
    elif mutation=='duplicate':report['trials'].append(copy.deepcopy(t))
    elif mutation=='implementation':report['implementation_version']='other'
    elif mutation=='best_index':t['best']['generation_index']=999
    elif mutation=='genes':t['best']['genes']=[]
    elif mutation=='density':t['best']['density']=0.99
    elif mutation=='counts':t['best']['placed_count']=999
    elif mutation=='extra':t['best']['extra']=None
    with pytest.raises((TypeError,ValueError)):validate_comparison_report(report)


@pytest.mark.parametrize(('path','metadata_version'), [
    (FIXTURES/'run_metadata_v2.json', 2),
    (FIXTURES/'comparison_v4.json', 2),
])
def test_execution_record_selection_supports_current_formats(path, metadata_version):
    records=read_execution_records(path)
    assert records
    assert all(record['version']==metadata_version for record in records)


@pytest.mark.parametrize('fixture_name', ['metadata_v2'])
def test_manual_replay_rejects_unknown_reference_version(request, fixture_name):
    record=request.getfixturevalue(fixture_name)
    record['decoder']['reference_version']=f"{record['decoder']['name']}-ref-v99"
    with pytest.raises(ValueError, match='reference_version'):
        prepare_manual_replay(record)


def test_v4_trial_replay_restores_settings_and_observes_cpu_once(tmp_path, monkeypatch):
    old=read_execution_records(FIXTURES/'comparison_v4.json')[0]
    original=old.to_dict()
    original['problem']['source']=str(tmp_path/'source-does-not-exist.txt')
    old=RunMetadata(original)
    before=old.to_dict()
    calls=[]
    resolutions=[]
    import packing.run_context as run_context
    original_resolver=run_context.resolve_workers

    def resolve_once(*args, **kwargs):
        resolutions.append('resolved')
        return original_resolver(*args, **kwargs)

    monkeypatch.setattr(run_context, 'resolve_workers', resolve_once)

    def read_cpu_once():
        calls.append('observed')
        return cpu(4)

    settings=prepare_manual_replay(old, cpu_reader=read_cpu_once)
    expected=DecoderConfig(
        candidate_policy=old['decoder']['candidate_policy'],
        candidate_policy_version=old['decoder']['candidate_policy_version'],
        min_support_percent=old['decoder']['min_support_percent'],
        support_model_version=old['decoder']['support_model_version'],
    )
    assert settings.decoder_config==expected
    assert [box.id for box in settings.boxes]==old['problem']['effective_box_order']
    assert calls==[]

    new=run_manual_replay(old, cpu_reader=read_cpu_once)
    assert calls==['observed']
    assert resolutions==['resolved']
    assert not (tmp_path/'source-does-not-exist.txt').exists()
    assert old.to_dict()==before
    assert new.run_metadata['run_id']!=old['run_id']
    assert new.run_metadata['replay_of']==old['run_id']
    assert new.run_metadata['version']==2
    assert new.run_metadata['decoder']['reference_version']==f"{old['decoder']['name']}-ref-v2"
    assert new.run_metadata['decoder']['candidate_policy']==expected.candidate_policy
    assert new.run_metadata['decoder']['candidate_policy_version']==expected.candidate_policy_version
    assert new.run_metadata['decoder']['min_support_percent']==expected.min_support_percent
    assert new.run_metadata['decoder']['support_model_version']==expected.support_model_version
    assert new.run_metadata['problem']==old['problem']
    assert new.run_metadata['execution']['cache']==old['execution']['cache']
    assert new.run_metadata['execution']['workers']['requested_mode']=='manual'
    assert new.run_metadata['execution']['workers']['resolved_workers']==old['execution']['workers']['resolved_workers']
    assert new.run_metadata['environment']['os']['system']==__import__('platform').system()


def test_windows_preflight_all_integer_entries(monkeypatch):
    import packing.workers as workers
    monkeypatch.setattr(workers.platform,'system',lambda:'Windows')
    assert GAExecutionConfig(61).workers==61
    with pytest.raises(ValueError,match='61'):GAExecutionConfig(62)


def test_source_definition_mismatch_preflight():
    with pytest.raises(ValueError,match='definitions'):
        run_bl_ga(Container(2,2,2),(Box(1,1,1,1),),config=GAConfig(1,1),
                  run_options=RunOptions(source_boxes=(Box(1,2,1,1),)))


def test_cpu_fallback_does_not_change_manual_or_require_label(monkeypatch):
    import packing.workers as workers
    monkeypatch.setattr(workers,'cpu_label',lambda:(None,None))
    run=run_bl_ga(Container(1,1,1),(),config=GAConfig(1,1),execution=GAExecutionConfig(2),
        run_options=RunOptions(cpu_reader=lambda:detect_cpu(SimpleNamespace())))
    assert run.execution.workers==2
    assert run.run_metadata['environment']['cpu']['label'] is None
    assert run.run_metadata['execution']['start_method']=='spawn'


def test_resolution_observer_cannot_mutate_snapshot():
    def observer(ex,env):
        ex['workers']['resolved_workers']=999
        env['cpu']['detection']['available_logical_estimate']=999
    run=run_bl_ga(Container(1,1,1),(),config=GAConfig(1,1),
        run_options=RunOptions(cpu_reader=lambda:cpu(1),resolution_callback=observer))
    assert run.execution.workers==1
    assert run.run_metadata['execution']['workers']['resolved_workers']==1
    assert run.run_metadata['environment']['cpu']['detection']['available_logical_estimate']==1


def test_depth_tie_order_and_unknown_ids(metadata):
    p=metadata['problem']
    p['source_boxes']=[[1,1,1,2],[2,1,1,2],[3,1,1,1]]
    p['source_box_order']=[1,2,3]
    p['order_mode']='depth_desc_stable'
    p['effective_box_order']=[2,1,3]
    with pytest.raises(ValueError):validate_run_metadata(metadata)
    p['effective_box_order']=[1,2,99]
    with pytest.raises(ValueError):validate_run_metadata(metadata)
    p['source_boxes'][1][0]=1
    with pytest.raises(ValueError):validate_run_metadata(metadata)


def test_benchmark_cli_and_compare_v4_atomic_writer(tmp_path):
    import subprocess
    fixture=tmp_path/'fixture.txt';fixture.write_text('{2,2,2}{1,1,1}')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
    directory=tmp_path/'runs'
    subprocess.run([sys.executable,str(ROOT/'benchmarks'/'bench_ga.py'),str(fixture),
                    '--population','1','--generations','1','--repeat','2','--output-dir',str(directory)],
                   env=env,check=True,capture_output=True)
    records=[read_run_metadata_json(p) for p in directory.glob('*.run.json')]
    assert len(records)==2 and records[0]['run_id']!=records[1]['run_id']
    assert all(m['version']==2 and m['decoder']['reference_version']=='BL-ref-v2' for m in records)
    assert all(m['origin']=='benchmark' and m['execution']['workers']['requested_mode']=='manual' for m in records)
    output=tmp_path/'comparison.json'
    subprocess.run([sys.executable,str(ROOT/'benchmarks'/'compare_ga.py'),str(fixture),
                    '--decoder','PP','--population','1','--generations','1','--output',str(output)],
                   env=env,check=True,capture_output=True)
    report=read_comparison_report_json(output)
    assert report['version']==4
    assert report['trials'][0]['run_metadata']['origin']=='compare_cli'
    assert report['trials'][0]['run_metadata']['version']==2
    assert set(report['trials'][0])=={'run_metadata','best'}


def test_comparison_writer_replace_failure_preserves_existing(tmp_path,monkeypatch):
    import packing.io_run_metadata as io
    report=read_comparison_report_json(FIXTURES/'comparison_v4.json')
    path=tmp_path/'comparison.json';path.write_text('keep')
    def fail(*_):raise OSError('replace')
    monkeypatch.setattr(io.os,'replace',fail)
    with pytest.raises(OSError):write_comparison_report_json(path,report)
    assert path.read_text()=='keep' and list(tmp_path.iterdir())==[path]
