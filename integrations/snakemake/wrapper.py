"""Minimal Snakemake wrapper for shiftmlwf predict."""

from __future__ import annotations

from pathlib import Path
import subprocess


def run_shiftmlwf(input_paths: list[str], output_dir: str, extra_args: list[str] | None = None) -> None:
    cmd = ["shiftmlwf", "predict", *input_paths, "--out", output_dir]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # Example standalone invocation.
    run_shiftmlwf(["examples/minimal.xyz"], "out/snakemake")
