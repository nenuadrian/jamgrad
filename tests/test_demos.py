import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMOS_DIR = PROJECT_ROOT / "demos"


def _run_demo(tmp_path, script_name, args=None, timeout=90):
    cmd = [sys.executable, str(DEMOS_DIR / script_name)]
    if args:
        cmd.extend(args)

    env = os.environ.copy()
    root_str = str(PROJECT_ROOT)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{root_str}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else root_str
    )

    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"Demo {script_name} failed.\n"
        f"Command: {' '.join(cmd)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


@pytest.mark.parametrize(
    ("script_name", "args"),
    [
        ("demo.py", []),
        ("demo_chain_rule.py", []),
        ("demo_eigenvalues.py", ["--matrix", "symmetric3", "--iters", "100"]),
        ("demo_nn.py", ["--epochs", "20", "--log-interval", "20", "--quiet"]),
    ],
)
def test_demo_smoke(tmp_path, script_name, args):
    _run_demo(tmp_path, script_name, args=args)


def test_demo_outputs_graph_file(tmp_path):
    _run_demo(tmp_path, "demo.py")
    assert (tmp_path / "computation_graph.dot").exists()


def test_demo_chain_rule_outputs_graph_file(tmp_path):
    _run_demo(tmp_path, "demo_chain_rule.py")
    assert (tmp_path / "chain_rule_graph.dot").exists()


def test_demo_mnist_smoke_opt_in(tmp_path):
    _run_demo(
        tmp_path,
        "demo_mnist.py",
        args=[
            "--dataset-size",
            "5000",
            "--epochs",
            "1",
            "--batch-size",
            "128",
            "--sample-predictions",
            "0",
            "--quiet",
        ],
        timeout=600,
    )
