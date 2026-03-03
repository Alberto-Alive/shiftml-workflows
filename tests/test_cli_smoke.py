from __future__ import annotations

import shutil
import subprocess
import sys


def test_python_module_info_command() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "shiftml_workflows", "info"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_help_command() -> None:
    shiftmlwf = shutil.which("shiftmlwf")
    if shiftmlwf:
        cmd = [shiftmlwf, "--help"]
    else:
        cmd = [sys.executable, "-m", "shiftml_workflows", "--help"]

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
