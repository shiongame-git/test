"""Static workers resolution and CPU observations, independent of GUI and GA RNG."""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from .source import source_revision

RESOLVER_VERSION = "workers-static-v1"
CPU_METHODS = ("os.process_cpu_count", "os.sched_getaffinity", "os.cpu_count", "fallback")


def positive_int(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


@dataclass(frozen=True, slots=True)
class WorkersRequest:
    mode: str = "manual"
    manual_workers: int | None = 1

    def __post_init__(self):
        if type(self.mode) is not str:
            raise TypeError("workers mode must be str")
        if self.mode not in ("auto", "max", "manual"):
            raise ValueError("unknown workers mode")
        if self.mode == "manual":
            positive_int(self.manual_workers, "manual_workers")
        elif self.manual_workers is not None:
            raise ValueError("Auto/Max manual_workers must be None")


def executor_limit(system=None):
    system = platform.system() if system is None else system
    return 61 if system == "Windows" else None


def preflight_workers(workers):
    positive_int(workers, "workers")
    limit = 61
    if workers > limit:
        raise ValueError(f"workers={workers} exceeds ProcessPoolExecutor limit {limit} on Windows")


def resolve_workers(request, available, population_size, limit=None):
    if type(request) is not WorkersRequest:
        raise TypeError("request must be WorkersRequest")
    a = positive_int(available, "available")
    p = positive_int(population_size, "population_size")
    if limit is not None:
        positive_int(limit, "limit")
    reasons, warnings = [], []
    if request.mode == "manual":
        resolved = request.manual_workers
        if limit is not None and resolved > limit:
            raise ValueError(f"manual workers={resolved} exceeds executor limit {limit}")
        if resolved > a:
            warnings.append("manual_exceeds_cpu_estimate")
        if resolved > p:
            warnings.append("manual_exceeds_population")
    else:
        target = max(1, (3 * a) // 4) if request.mode == "auto" else a
        caps = ([('auto_cap', 8)] if request.mode == "auto" else []) + [('population_size', p)]
        if limit is not None:
            caps.append(('executor_limit', limit))
        resolved = min(target, *(value for _, value in caps))
        if resolved < target:
            reasons = [name for name, value in caps if value == resolved]
    return dict(requested_mode=request.mode, manual_workers=request.manual_workers,
                resolved_workers=resolved, resolver_version=RESOLVER_VERSION,
                auto_ratio=0.75, auto_cap=8, limit_reasons=reasons, warning_reasons=warnings)


def detect_cpu(api=None):
    """Inject an os-like API for deterministic tests. Stop after the first success."""
    api = os if api is None else api
    attempts = []
    selected, available = None, None
    for method in CPU_METHODS:
        status, value, detail = "not_called", None, None
        if selected is None:
            if method == "fallback":
                status, value = "fallback", 1
            else:
                name = method.split('.')[1]
                fn = None if name == "sched_getaffinity" else getattr(api, name, None)
                if fn is None:
                    status = "unavailable"
                else:
                    try:
                        raw = count = fn()
                        if type(count) is int and count > 0:
                            status, value = "success", count
                        elif raw is None:
                            status = "none"
                        else:
                            status = "invalid"
                            value = count if type(count) is int else None
                            if value is None:
                                detail = f"{type(raw).__name__}: {raw!r}"[:2048]
                    except NotImplementedError as exc:
                        status, detail = "not_implemented", str(exc)[:2048]
                    except OSError as exc:
                        status, detail = "os_error", str(exc)[:2048]
            if status in ("success", "fallback"):
                selected, available = method, value
        attempts.append(dict(method=method, status=status, value=value, detail=detail))
    fallback = [] if selected == CPU_METHODS[0] else ["cpu_fallback" if selected == "fallback" else "cpu_api_fallback"]
    return dict(attempts=attempts, selected_method=selected,
                available_logical_estimate=available, fallback_reasons=fallback)


def cpu_label():
    try:
        label = platform.processor().strip()
        if label:
            return label, "platform.processor"
    except (OSError, NotImplementedError):
        pass
    label = os.environ.get('PROCESSOR_IDENTIFIER', '').strip()
    return (label, "PROCESSOR_IDENTIFIER") if label else (None, None)


def observe_environment(*, implementation_version, revision=None, cpu_reader=detect_cpu):
    system = platform.system() or "unknown"
    limit = executor_limit(system)
    label, label_source = cpu_label()
    return dict(os=dict(system=system, release=platform.release() or None, machine=platform.machine() or None),
                python=dict(implementation=platform.python_implementation(), version=platform.python_version()),
                implementation=dict(version=implementation_version, revision=source_revision() if revision is None else revision),
                executor=dict(name="concurrent.futures.ProcessPoolExecutor", worker_limit=limit,
                              limit_source="windows_process_pool_61" if limit else "no_known_limit"),
                cpu=dict(label=label, label_source=label_source, detection=cpu_reader()))
