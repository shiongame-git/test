"""Strict, atomic run metadata v2 and comparison report v4 I/O."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from ._io_integer import parse_public_decimal_token, public_int_text, require_public_io_int
from .errors import InputFormatError
from .io_result_json import _pairs_no_duplicates, _reject_constant
from .workers import CPU_METHODS, WorkersRequest, executor_limit, resolve_workers

ROOT = ('format', 'version', 'run_id', 'status', 'origin', 'replay_of', 'problem', 'decoder', 'ga', 'execution', 'environment', 'statistics')
PROBLEM = ('container', 'source_boxes', 'source_box_order', 'effective_box_order', 'order_mode', 'source', 'fixture_sha256')
DECODER_V2 = ('name', 'priority', 'reference_version', 'fitness_version', 'pp_initialization',
              'candidate_policy', 'candidate_policy_version', 'min_support_percent', 'support_model_version')
GA = ('population_size', 'generation_count', 'crossover_rate', 'mutation_rate', 'seed')
WORKERS = ('requested_mode', 'manual_workers', 'resolved_workers', 'resolver_version', 'auto_ratio', 'auto_cap', 'limit_reasons', 'warning_reasons')
STATS = ('requested', 'computed', 'cache_hits', 'elapsed_seconds', 'elapsed_scope')
BEST = ('generation_index', 'population_index', 'genes', 'fitness', 'placed_count', 'failed_count', 'placed_volume', 'density')


def obj(v, keys, path):
    if type(v) is not dict:
        raise TypeError(f'{path}: expected object')
    if set(v) != set(keys):
        raise ValueError(f'{path}: missing or unknown keys; expected {keys}')
    return v


def integer(v, path, minimum=0):
    require_public_io_int(v, name=path)
    if v < minimum:
        raise ValueError(f'{path}: integer below minimum {minimum}')
    return v


def number(v, path, minimum=0, maximum=None):
    if type(v) not in (int, float):
        raise TypeError(f'{path}: expected number, not boolean')
    if type(v) is int:
        require_public_io_int(v, name=path)
    try:
        finite = math.isfinite(v)
    except OverflowError:
        finite = False
    if not finite or v < minimum or (maximum is not None and v > maximum):
        raise ValueError(f'{path}: nonfinite or out-of-range number')
    return v


def string(v, path, nullable=False):
    if nullable and v is None:
        return
    if type(v) is not str:
        raise TypeError(f'{path}: expected string')
    if not v:
        raise ValueError(f'{path}: empty string')


def enum(v, choices, path):
    if type(v) is not str:
        raise TypeError(f'{path}: expected string enum')
    if v not in choices:
        raise ValueError(f'{path}: unsupported value')


def array(v, path):
    if type(v) is not list:
        raise TypeError(f'{path}: expected array')
    return v


def equal(v, expected, path):
    if v != expected:
        raise ValueError(f'{path}: inconsistent value')


def uuid4(v, path):
    string(v, path)
    try:
        parsed = UUID(v)
    except ValueError as exc:
        raise ValueError(f'{path}: invalid UUID') from exc
    if parsed.version != 4 or str(parsed) != v:
        raise ValueError(f'{path}: expected canonical lowercase UUID v4')


def validate_problem(p):
    obj(p, PROBLEM, '$.problem')
    c = array(p['container'], '$.problem.container')
    if len(c) != 3:
        raise ValueError('$.problem.container: expected three dimensions')
    for v in c:
        integer(v, '$.problem.container[]', 1)
    boxes = array(p['source_boxes'], '$.problem.source_boxes')
    ids = []
    for i, b in enumerate(boxes):
        array(b, f'$.problem.source_boxes[{i}]')
        if len(b) != 4:
            raise ValueError('$.problem.source_boxes[]: expected [id,w,h,d]')
        for j, v in enumerate(b):
            integer(v, f'$.problem.source_boxes[{i}][{j}]', 0 if j == 0 else 1)
        ids.append(b[0])
    if len(set(ids)) != len(ids):
        raise ValueError('$.problem.source_boxes: duplicate IDs')
    for key in ('source_box_order', 'effective_box_order'):
        for v in array(p[key], '$.problem.' + key):
            integer(v, '$.problem.' + key + '[]')
    equal(p['source_box_order'], ids, '$.problem.source_box_order')
    enum(p['order_mode'], ('source', 'depth_desc_stable'), '$.problem.order_mode')
    expected = ids if p['order_mode'] == 'source' else [b[0] for b in sorted(boxes, key=lambda b: -b[3])]
    equal(p['effective_box_order'], expected, '$.problem.effective_box_order')
    string(p['source'], '$.problem.source', True)
    if p['fixture_sha256'] is not None:
        string(p['fixture_sha256'], '$.problem.fixture_sha256')
        if not re.fullmatch('[0-9a-f]{64}', p['fixture_sha256']):
            raise ValueError('$.problem.fixture_sha256: invalid hash')


def validate_ga(g):
    obj(g, GA, '$.ga')
    integer(g['population_size'], '$.ga.population_size', 1)
    integer(g['generation_count'], '$.ga.generation_count', 1)
    integer(g['seed'], '$.ga.seed')
    for key in ('crossover_rate', 'mutation_rate'):
        number(g[key], '$.ga.' + key, maximum=1)


def validate_detection(d):
    obj(d, ('attempts', 'selected_method', 'available_logical_estimate', 'fallback_reasons'), '$.environment.cpu.detection')
    attempts = array(d['attempts'], '$.environment.cpu.detection.attempts')
    if len(attempts) != 4:
        raise ValueError('$.environment.cpu.detection.attempts: expected four attempts')
    selected, value = None, None
    failures = ('unavailable', 'none', 'invalid', 'os_error', 'not_implemented')
    for i, (a, method) in enumerate(zip(attempts, CPU_METHODS)):
        path = f'$.environment.cpu.detection.attempts[{i}]'
        obj(a, ('method', 'status', 'value', 'detail'), path)
        enum(a['method'], (method,), path + '.method')
        enum(a['status'], (*failures, 'success', 'not_called', 'fallback'), path + '.status')
        if a['value'] is not None:
            require_public_io_int(a['value'], name=path + '.value')
        if a['detail'] is not None:
            if type(a['detail']) is not str:
                raise TypeError(path + '.detail: expected string or null')
            if len(a['detail']) > 2048:
                raise ValueError(path + '.detail: over 2048 characters')
        status = a['status']
        if selected is not None:
            equal(status, 'not_called', path + '.status')
            equal(a['value'], None, path + '.value')
            equal(a['detail'], None, path + '.detail')
        elif i == 3:
            equal(status, 'fallback', path + '.status')
            equal(a['value'], 1, path + '.value')
            equal(a['detail'], None, path + '.detail')
            selected, value = method, 1
        elif status == 'success':
            integer(a['value'], path + '.value', 1)
            equal(a['detail'], None, path + '.detail')
            selected, value = method, a['value']
        elif status in failures:
            if status == 'none' and i == 1:
                raise ValueError(path + ': affinity cannot use none status')
            if status == 'invalid':
                if a['value'] is not None and a['value'] > 0:
                    raise ValueError(path + ': invalid positive integer')
                if a['value'] is None and a['detail'] is None:
                    raise ValueError(path + ': invalid noninteger requires diagnostic')
            else:
                equal(a['value'], None, path + '.value')
        else:
            raise ValueError(path + ': attempt skipped before success')
    string(d['selected_method'], '$.environment.cpu.detection.selected_method')
    equal(d['selected_method'], selected, '$.environment.cpu.detection.selected_method')
    integer(d['available_logical_estimate'], '$.environment.cpu.detection.available_logical_estimate', 1)
    equal(d['available_logical_estimate'], value, '$.environment.cpu.detection.available_logical_estimate')
    array(d['fallback_reasons'], '$.environment.cpu.detection.fallback_reasons')
    for reason in d['fallback_reasons']:
        enum(reason, ('cpu_fallback', 'cpu_api_fallback'), '$.environment.cpu.detection.fallback_reasons[]')
    expected = [] if selected == CPU_METHODS[0] else ['cpu_fallback' if selected == 'fallback' else 'cpu_api_fallback']
    equal(d['fallback_reasons'], expected, '$.environment.cpu.detection.fallback_reasons')


def validate_environment(e):
    obj(e, ('os', 'python', 'implementation', 'executor', 'cpu'), '$.environment')
    for name, keys in [('os', ('system', 'release', 'machine')), ('python', ('implementation', 'version')),
                       ('implementation', ('version', 'revision'))]:
        obj(e[name], keys, '$.environment.' + name)
        for key in keys:
            string(e[name][key], '$.environment.' + name + '.' + key,
                   key in ('release', 'machine', 'revision'))
    ex = obj(e['executor'], ('name', 'worker_limit', 'limit_source'), '$.environment.executor')
    enum(ex['name'], ('concurrent.futures.ProcessPoolExecutor',), '$.environment.executor.name')
    if ex['worker_limit'] is not None:
        integer(ex['worker_limit'], '$.environment.executor.worker_limit', 1)
    equal(ex['worker_limit'], executor_limit(e['os']['system']), '$.environment.executor.worker_limit')
    enum(ex['limit_source'], ('windows_process_pool_61' if e['os']['system'] == 'Windows' else 'no_known_limit',), '$.environment.executor.limit_source')
    cpu = obj(e['cpu'], ('label', 'label_source', 'detection'), '$.environment.cpu')
    string(cpu['label'], '$.environment.cpu.label', True)
    if cpu['label'] is None:
        equal(cpu['label_source'], None, '$.environment.cpu.label_source')
    else:
        enum(cpu['label_source'], ('platform.processor', 'proc_cpuinfo', 'PROCESSOR_IDENTIFIER'), '$.environment.cpu.label_source')
    validate_detection(cpu['detection'])


def validate_run_metadata(metadata):
    m = metadata.to_dict() if type(metadata) is RunMetadata else metadata
    obj(m, ROOT, '$')
    enum(m['format'], ('packing-ga-run-metadata',), '$.format')
    integer(m['version'], '$.version')
    if m['version'] != 2:
        raise ValueError('$.version: unsupported run metadata version')
    uuid4(m['run_id'], '$.run_id')
    enum(m['status'], ('completed',), '$.status')
    enum(m['origin'], ('gui', 'api', 'compare_cli', 'benchmark'), '$.origin')
    if m['replay_of'] is not None:
        uuid4(m['replay_of'], '$.replay_of')
        if m['replay_of'] == m['run_id']:
            raise ValueError('$.replay_of: cannot reference itself')
    validate_problem(m['problem'])
    d = obj(m['decoder'], DECODER_V2, '$.decoder')
    enum(d['name'], ('BL', 'PP'), '$.decoder.name')
    enum(d['priority'], ('BottomFirst', 'BackFirst'), '$.decoder.priority')
    enum(d['reference_version'], (d["name"] + '-ref-v2',), '$.decoder.reference_version')
    enum(d['fitness_version'], ('density-v1',), '$.decoder.fitness_version')
    equal(d['pp_initialization'], None if d['name'] == 'BL' else 'bl_first_generation', '$.decoder.pp_initialization')
    enum(d['candidate_policy'], ('Cartesian', 'LocalNeighbors'), '$.decoder.candidate_policy')
    equal(integer(d['candidate_policy_version'], '$.decoder.candidate_policy_version'), 1,
          '$.decoder.candidate_policy_version')
    integer(d['min_support_percent'], '$.decoder.min_support_percent')
    if d['min_support_percent'] > 100:
        raise ValueError('$.decoder.min_support_percent: expected integer from 0 through 100')
    enum(d['support_model_version'], ('support-union-v1',), '$.decoder.support_model_version')
    validate_ga(m['ga'])
    validate_environment(m['environment'])
    ex = obj(m['execution'], ('workers', 'cache', 'start_method'), '$.execution')
    if type(ex['cache']) is not bool:
        raise TypeError('$.execution.cache: expected boolean')
    w = obj(ex['workers'], WORKERS, '$.execution.workers')
    enum(w['requested_mode'], ('auto', 'max', 'manual'), '$.execution.workers.requested_mode')
    if w['manual_workers'] is not None:
        integer(w['manual_workers'], '$.execution.workers.manual_workers', 1)
    integer(w['resolved_workers'], '$.execution.workers.resolved_workers', 1)
    enum(w['resolver_version'], ('workers-static-v1',), '$.execution.workers.resolver_version')
    number(w['auto_ratio'], '$.execution.workers.auto_ratio')
    integer(w['auto_cap'], '$.execution.workers.auto_cap', 1)
    for key in ('limit_reasons', 'warning_reasons'):
        array(w[key], '$.execution.workers.' + key)
        choices = ('auto_cap', 'population_size', 'executor_limit') if key == 'limit_reasons' else ('manual_exceeds_cpu_estimate', 'manual_exceeds_population')
        for reason in w[key]:
            enum(reason, choices, '$.execution.workers.' + key + '[]')
    expected = resolve_workers(WorkersRequest(w['requested_mode'], w['manual_workers']),
                               m['environment']['cpu']['detection']['available_logical_estimate'],
                               m['ga']['population_size'], m['environment']['executor']['worker_limit'])
    equal(w, expected, '$.execution.workers')
    equal(ex['start_method'], None if w['resolved_workers'] == 1 else 'spawn', '$.execution.start_method')
    if m['replay_of'] is not None and w['requested_mode'] != 'manual':
        raise ValueError('$.replay_of: requires Manual request')
    stats = obj(m['statistics'], STATS, '$.statistics')
    for key in ('requested', 'computed', 'cache_hits'):
        integer(stats[key], '$.statistics.' + key)
    number(stats['elapsed_seconds'], '$.statistics.elapsed_seconds')
    enum(stats['elapsed_scope'], ('ga_call_including_resolution_and_pool',), '$.statistics.elapsed_scope')
    equal(stats['requested'], m['ga']['population_size'] * m['ga']['generation_count'], '$.statistics.requested')
    equal(stats['computed'] + stats['cache_hits'], stats['requested'], '$.statistics.computed+cache_hits')
    if not ex['cache']:
        equal(stats['cache_hits'], 0, '$.statistics.cache_hits')


def _ordered(m):
    out = {k: m[k] for k in ROOT}
    for key, keys in [('problem', PROBLEM), ('ga', GA), ('statistics', STATS)]:
        out[key] = {k: m[key][k] for k in keys}
    out['decoder'] = {k: m['decoder'][k] for k in DECODER_V2}
    out['execution'] = {k: m['execution'][k] for k in ('workers', 'cache', 'start_method')}
    out['execution']['workers'] = {k: m['execution']['workers'][k] for k in WORKERS}
    e = m['environment']
    out['environment'] = {name: {k: e[name][k] for k in keys} for name, keys in [
        ('os', ('system', 'release', 'machine')), ('python', ('implementation', 'version')),
        ('implementation', ('version', 'revision')), ('executor', ('name', 'worker_limit', 'limit_source')),
        ('cpu', ('label', 'label_source', 'detection'))]}
    d = e['cpu']['detection']
    out['environment']['cpu']['detection'] = {k: d[k] for k in ('attempts', 'selected_method', 'available_logical_estimate', 'fallback_reasons')}
    out['environment']['cpu']['detection']['attempts'] = [{k: a[k] for k in ('method', 'status', 'value', 'detail')} for a in d['attempts']]
    return out


def render_json(v, level=0):
    """Bounded integer rendering also works under a lower CPython digit guard."""
    if type(v) is int:
        return public_int_text(v, name='$')
    if type(v) in (dict, list):
        if not v:
            return '{}' if type(v) is dict else '[]'
        indent, inner = '  ' * level, '  ' * (level + 1)
        if type(v) is dict:
            parts = [inner + json.dumps(k, ensure_ascii=False) + ': ' + render_json(x, level + 1) for k, x in v.items()]
            return '{\n' + ',\n'.join(parts) + '\n' + indent + '}'
        return '[\n' + ',\n'.join(inner + render_json(x, level + 1) for x in v) + '\n' + indent + ']'
    return json.dumps(v, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True, slots=True, init=False)
class RunMetadata(Mapping):
    """Immutable value snapshot. Returned nested objects are independent copies."""
    _text: str

    def __init__(self, data):
        validate_run_metadata(data)
        object.__setattr__(self, '_text', render_json(_ordered(data)))

    def to_dict(self):
        return json.loads(self._text, parse_int=parse_public_decimal_token)

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __iter__(self):
        return iter(ROOT)

    def __len__(self):
        return len(ROOT)


def load_json(path):
    p = Path(path)
    try:
        return json.loads(p.read_bytes().decode('utf-8-sig'), object_pairs_hook=_pairs_no_duplicates,
                          parse_constant=_reject_constant, parse_int=parse_public_decimal_token)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise InputFormatError(f'{p}: {exc}') from exc


def atomic_write(path, data):
    p = Path(path)
    text = render_json(data) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=p.parent, prefix='.' + p.name + '.', suffix='.tmp', delete=False) as f:
            temporary = f.name
            f.write(text)
        os.replace(temporary, p)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def read_run_metadata_json(path):
    try:
        return RunMetadata(load_json(path))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InputFormatError):
            raise
        raise InputFormatError(f'{path}: {exc}') from exc


def write_run_metadata_json(path, metadata):
    validate_run_metadata(metadata)
    data = metadata.to_dict() if type(metadata) is RunMetadata else metadata
    atomic_write(path, _ordered(data))


def validate_best(best, problem, ga):
    obj(best, BEST, '$.trials[].best')
    for key in ('generation_index', 'population_index', 'placed_count', 'failed_count', 'placed_volume'):
        integer(best[key], '$.trials[].best.' + key)
    if best['generation_index'] >= ga['generation_count'] or best['population_index'] >= ga['population_size']:
        raise ValueError('$.trials[].best: index out of range')
    genes = array(best['genes'], '$.trials[].best.genes')
    for v in genes:
        integer(v, '$.trials[].best.genes[]')
    ids = problem['source_box_order']
    if len(genes) != len(ids) or set(genes) != set(ids):
        raise ValueError('$.trials[].best.genes: not a full permutation')
    for key in ('fitness', 'density'):
        number(best[key], '$.trials[].best.' + key, maximum=1)
    equal(best['fitness'], best['density'], '$.trials[].best.fitness')
    equal(best['placed_count'] + best['failed_count'], len(ids), '$.trials[].best.counts')
    volume = math.prod(problem['container'])
    if best['placed_volume'] > min(volume, sum(math.prod(b[1:]) for b in problem['source_boxes'])):
        raise ValueError('$.trials[].best.placed_volume: out of range')
    equal(best['density'], best['placed_volume'] / volume, '$.trials[].best.density')


def validate_comparison_report(report):
    obj(report, ('format', 'version', 'implementation_version', 'trials'), '$')
    enum(report['format'], ('packing3d-comparison-report',), '$.format')
    integer(report['version'], '$.version')
    if report['version'] != 4:
        raise ValueError('$.version: unsupported comparison version')
    string(report['implementation_version'], '$.implementation_version')
    trials = array(report['trials'], '$.trials')
    ids = set()
    for t in trials:
        obj(t, ('run_metadata', 'best'), '$.trials[]')
        m = t['run_metadata']
        validate_run_metadata(m)
        equal(m['version'], 2,
              '$.trials[].run_metadata.version')
        equal(m['environment']['implementation']['version'], report['implementation_version'], '$.implementation_version')
        if m['run_id'] in ids:
            raise ValueError('$.trials: duplicate run_id')
        ids.add(m['run_id'])
        validate_best(t['best'], m['problem'], m['ga'])


def read_comparison_report_json(path):
    report = load_json(path)
    try:
        validate_comparison_report(report)
    except (TypeError, ValueError) as exc:
        raise InputFormatError(f'{path}: {exc}') from exc
    return report


def write_comparison_report_json(path, report):
    validate_comparison_report(report)
    ordered = {k: report[k] for k in ('format', 'version', 'implementation_version', 'trials')}
    ordered['trials'] = [dict(run_metadata=_ordered(t['run_metadata']),
                              best={k: t['best'][k] for k in BEST})
                         for t in report['trials']]
    atomic_write(path, ordered)


def read_execution_records(path):
    """Return replay records from metadata v2 or comparison v4."""
    data = load_json(path)
    try:
        if type(data) is dict and data.get('format') == 'packing-ga-run-metadata':
            return (RunMetadata(data),)
        validate_comparison_report(data)
        return tuple(RunMetadata(t['run_metadata']) for t in data['trials'])
    except (TypeError, ValueError) as exc:
        raise InputFormatError(f'{path}: {exc}') from exc
