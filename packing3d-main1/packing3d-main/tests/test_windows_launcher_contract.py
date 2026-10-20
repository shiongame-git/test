from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_windows_launcher_monitors_uv_exit_ready_marker_and_logs():
    bat_bytes = (ROOT / "run_gui.bat").read_bytes()
    ps1_bytes = (ROOT / "demo" / "windows_launcher.ps1").read_bytes()
    bat = bat_bytes.decode("ascii")
    ps1 = ps1_bytes.decode("utf-8-sig")

    # cmd.exe parses the batch file before any in-script code-page change can
    # reliably rescue non-ASCII source text. Keep the BAT itself ASCII-only
    # and use CRLF so CP932/UTF-8 differences cannot corrupt command tokens.
    assert all(byte < 0x80 for byte in bat_bytes)
    assert b"\n" not in bat_bytes.replace(b"\r\n", b"")
    assert "chcp " not in bat.lower()
    assert "where uv" not in bat.lower()
    assert "where powershell.exe" in bat.lower()
    assert "%~dp0demo\\windows_launcher.ps1" in bat
    assert 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo\\windows_launcher.ps1"' in bat

    # Windows PowerShell 5.1 treats BOM-less UTF-8 scripts as the legacy ANSI
    # code page. A UTF-8 BOM preserves the Japanese diagnostic strings.
    assert ps1_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\n" not in ps1_bytes.replace(b"\r\n", b"")

    assert "[switch]$Monitor" in ps1
    assert "[switch]$RunGuiChild" in ps1
    assert "[string]$ExitCodeFile" in ps1
    assert "$PSCommandPath" in ps1
    assert "-Monitor" in ps1
    assert "PACKING3D_GUI_READY_FILE" in ps1
    assert "RedirectStandardError" in ps1
    assert "RedirectStandardOutput" in ps1
    assert "$process.ExitCode" not in ps1
    assert "PACKING3D_GUI_READY_FILE" in ps1
    assert "packing3d-gui-exit-" in ps1
    assert "Set-Content -LiteralPath $ExitCodeFile" in ps1
    assert "[int]::TryParse" in ps1
    assert "if ($null -eq $exitCode)" in ps1
    assert "$startupReady" in ps1
    assert '"demo\\gui.pyw"' in ps1
    assert 'run --locked python "run.py" "demo\\gui.pyw"' in ps1
    assert "Show-LaunchFailure" in ps1


def test_gui_ready_marker_is_headless_and_does_not_require_tk_window(tmp_path: Path, monkeypatch):
    pytest.importorskip("tkinter")
    demo_dir = ROOT / "demo"
    monkeypatch.syspath_prepend(str(demo_dir))
    loader = importlib.machinery.SourceFileLoader("packing3d_demo_gui", str(demo_dir / "gui.pyw"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    marker = tmp_path / "ready.txt"
    monkeypatch.setenv("PACKING3D_GUI_READY_FILE", str(marker))
    module._notify_launcher_ready()
    assert marker.read_text(encoding="ascii") == "ready\n"


def test_readme_parallel_example_runs_with_real_spawn(tmp_path: Path):
    import subprocess

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "packing3d" in readme
    assert "import名は`packing`" in readme
    blocks = readme.split("```python")
    matching = []
    for tail in blocks[1:]:
        code, marker, _ = tail.partition("```")
        if marker and "GAExecutionConfig(workers=2, cache=True)" in code:
            matching.append(code.strip() + "\n")
    assert len(matching) == 1
    assert 'if __name__ == "__main__":' in matching[0]

    script = tmp_path / "readme_parallel_example.py"
    script.write_text(matching[0], encoding="utf-8", newline="\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 3
    assert all(line.strip().isdigit() for line in lines)
