from __future__ import annotations

import os
import time
from pathlib import Path


def _mark_gui_spawn_worker() -> None:
    marker_dir = os.environ.get("PACKING3D_GUI_SPAWN_MARKER_DIR")
    if marker_dir is not None:
        directory = Path(marker_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{os.getpid()}.marker").write_text("spawned", encoding="utf-8")


def delayed_failure_bl_worker(container, boxes, individual, priority, decoder_config, *_):
    """Fail index-shaped individuals in deliberately reversed completion order."""
    if individual.genes and individual.genes[0] == 1:
        time.sleep(0.25)
        raise TimeoutError("low-index delayed failure")
    time.sleep(0.01)
    raise RuntimeError("high-index early failure")


def gui_spawn_bl_worker(*args):
    _mark_gui_spawn_worker()
    from packing.execution import _bl_worker

    return _bl_worker(*args)


def gui_spawn_pp_worker(*args):
    _mark_gui_spawn_worker()
    from packing.execution import _pp_worker

    return _pp_worker(*args)


def gui_spawn_failing_bl_worker(*args):
    _mark_gui_spawn_worker()
    raise RuntimeError("injected GUI BL spawn worker failure")


def gui_spawn_failing_pp_worker(*args):
    _mark_gui_spawn_worker()
    raise RuntimeError("injected GUI PP spawn worker failure")
