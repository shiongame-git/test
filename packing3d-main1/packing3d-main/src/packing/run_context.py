"""One parent-thread snapshot shared by every GA entry point."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from ._decoder_config import (
    DecoderConfig,
    resolve_decoder_config,
)
from .execution import GAExecutionConfig
from .io_run_metadata import RunMetadata, validate_run_metadata
from .workers import WorkersRequest, detect_cpu, observe_environment, preflight_workers, resolve_workers

from .source import SOURCE_LABEL, source_revision


@dataclass(frozen=True, slots=True)
class RunOptions:
    request: WorkersRequest | None = None
    origin: str = 'api'
    replay_of: str | None = None
    source_boxes: tuple | None = None
    order_mode: str = 'source'
    source: str | None = None
    fixture_sha256: str | None = None
    implementation_version: str = SOURCE_LABEL
    revision: str | None = None
    cpu_reader: Callable = detect_cpu
    resolution_callback: Callable | None = None


@dataclass(frozen=True, slots=True)
class RunContext:
    started: float
    execution: GAExecutionConfig
    prototype: RunMetadata

    def complete(self, stats):
        data = self.prototype.to_dict()
        data['statistics'].update(requested=stats.requested, computed=stats.computed, cache_hits=stats.cache_hits,
                                  elapsed_seconds=time.perf_counter() - self.started)
        return RunMetadata(data)


def begin_run(
    container,
    boxes,
    decoder,
    priority,
    config,
    seed,
    execution,
    decoder_config,
    options=None,
):
    decoder_config = resolve_decoder_config(decoder_config)
    options = RunOptions() if options is None else options
    if type(options) is not RunOptions:
        raise TypeError('run_options must be RunOptions')
    started = time.perf_counter()
    environment = observe_environment(implementation_version=options.implementation_version,
                                      revision=options.revision, cpu_reader=options.cpu_reader)
    request = options.request or WorkersRequest('manual', execution.workers)
    workers = resolve_workers(request, environment['cpu']['detection']['available_logical_estimate'],
                              config.population_size, environment['executor']['worker_limit'])
    resolved = GAExecutionConfig(workers['resolved_workers'], execution.cache)
    source = boxes if options.source_boxes is None else options.source_boxes
    if type(source) is not tuple:
        raise TypeError('source_boxes must be a tuple')
    from .model import Box
    if any(type(b) is not Box for b in source):
        raise TypeError('source_boxes must contain the same Box definitions as execution input')
    if {b.id: b for b in source} != {b.id: b for b in boxes}:
        raise ValueError('source_boxes definitions differ from execution input')
    data = dict(format='packing-ga-run-metadata', version=2, run_id=str(uuid4()), status='completed',
                origin=options.origin, replay_of=options.replay_of,
                problem=dict(container=[container.W, container.H, container.D],
                             source_boxes=[[b.id, b.w, b.h, b.d] for b in source],
                             source_box_order=[b.id for b in source], effective_box_order=[b.id for b in boxes],
                             order_mode=options.order_mode, source=options.source, fixture_sha256=options.fixture_sha256),
                decoder=dict(name=decoder, priority=priority.value, reference_version=decoder + '-ref-v2',
                             fitness_version='density-v1', pp_initialization=None if decoder == 'BL' else 'bl_first_generation',
                             candidate_policy=decoder_config.candidate_policy,
                             candidate_policy_version=decoder_config.candidate_policy_version,
                             min_support_percent=decoder_config.min_support_percent,
                             support_model_version=decoder_config.support_model_version),
                ga=dict(population_size=config.population_size, generation_count=config.generation_count,
                        crossover_rate=config.crossover_rate, mutation_rate=config.mutation_rate, seed=seed),
                execution=dict(workers=workers, cache=resolved.cache, start_method=None if resolved.workers == 1 else 'spawn'),
                environment=environment,
                statistics=dict(requested=config.population_size * config.generation_count,
                                computed=config.population_size * config.generation_count, cache_hits=0,
                                elapsed_seconds=0.0, elapsed_scope='ga_call_including_resolution_and_pool'))
    # Validate all supplied options before creating an executor or consuming GA RNG.
    prototype = RunMetadata(data)
    if options.resolution_callback is not None:
        options.resolution_callback(prototype['execution'], prototype['environment'])
    return RunContext(started, resolved, prototype)


@dataclass(frozen=True, slots=True)
class ReplaySettings:
    container: object
    boxes: tuple
    decoder: str
    priority: object
    config: object
    seed: int
    execution: GAExecutionConfig
    decoder_config: DecoderConfig
    options: RunOptions
    environment_differences: tuple[str, ...]


def prepare_manual_replay(metadata, *, origin='api', cpu_reader=detect_cpu):
    """Validate and prepare; never execute or open the recorded source path."""
    from .bl import BLPriority
    from .ga import GAConfig
    from .model import Box, Container
    import platform

    validate_run_metadata(metadata)
    m = metadata.to_dict() if type(metadata) is RunMetadata else RunMetadata(metadata).to_dict()
    w = m['execution']['workers']['resolved_workers']
    preflight_workers(w)
    p, g = m['problem'], m['ga']
    d = m['decoder']
    decoder_config = DecoderConfig(
        candidate_policy=d['candidate_policy'],
        candidate_policy_version=d['candidate_policy_version'],
        min_support_percent=d['min_support_percent'],
        support_model_version=d['support_model_version'],
    )
    source = tuple(Box(*b) for b in p['source_boxes'])
    by_id = {b.id: b for b in source}
    boxes = tuple(by_id[i] for i in p['effective_box_order'])
    differences = []
    current = {'os': platform.system(), 'os.release': platform.release() or None, 'os.machine': platform.machine() or None,
               'python': platform.python_version(), 'python.implementation': platform.python_implementation(),
               'implementation': SOURCE_LABEL}
    previous = {'os': m['environment']['os']['system'], 'os.release': m['environment']['os']['release'],
                'os.machine': m['environment']['os']['machine'], 'python': m['environment']['python']['version'],
                'python.implementation': m['environment']['python']['implementation'],
                'implementation': m['environment']['implementation']['version']}
    for key in current:
        if previous[key] != current[key]:
            differences.append(f'{key}: {previous[key]} → {current[key]}')
    differences.append('implementation revision: ' + (m['environment']['implementation']['revision'] or 'unknown') + '; current revision: ' + (source_revision() or 'unknown') + '; exact reproduction is not verified')
    return ReplaySettings(Container(*p['container']), boxes, m['decoder']['name'], BLPriority(m['decoder']['priority']),
                          GAConfig(g['population_size'], g['generation_count'], float(g['crossover_rate']), float(g['mutation_rate'])),
                          g['seed'], GAExecutionConfig(w, m['execution']['cache']),
                          decoder_config,
                          RunOptions(request=WorkersRequest('manual', w), origin=origin, replay_of=m['run_id'],
                                     cpu_reader=cpu_reader,
                                     source_boxes=source, order_mode=p['order_mode'], source=p['source'], fixture_sha256=p['fixture_sha256']),
                          tuple(differences))


def run_manual_replay(metadata, *, origin='api', progress_callback=None, cpu_reader=detect_cpu):
    settings = prepare_manual_replay(metadata, origin=origin, cpu_reader=cpu_reader)
    from .ga import run_bl_ga
    from .pp_ga import run_pp_ga
    runner = run_bl_ga if settings.decoder == 'BL' else run_pp_ga
    return runner(settings.container, settings.boxes, priority=settings.priority, config=settings.config,
                  seed=settings.seed, execution=settings.execution, decoder_config=settings.decoder_config,
                  run_options=settings.options,
                  progress_callback=progress_callback)
