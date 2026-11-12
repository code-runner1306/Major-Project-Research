"""Opt-in end-to-end Windows launcher test with real CPU workers and artifacts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest


@pytest.mark.skipif(sys.platform != 'win32' or os.environ.get('RUN_48H_LAUNCHER_SMOKE') != '1',
                    reason='Set RUN_48H_LAUNCHER_SMOKE=1 on Windows to run the full launcher smoke')
def test_full_launcher_smoke():
    repo = Path(__file__).resolve().parents[1]
    relative_root = Path('output') / ('launcher_test_' + uuid4().hex)
    root = repo / relative_root
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    try:
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
             str(repo / 'run_48h_exploratory_study.ps1'), '-Family', 'both', '-SmokeTest',
             '-Python', sys.executable, '-OutputRoot', str(relative_root)],
            cwd=repo, env=env, capture_output=True, text=True, timeout=900,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((root / 'run_summary.json').read_text(encoding='utf-8-sig'))
        assert summary['status'] == 'complete' and summary['run_count'] == 8
        assert summary['training_barrier_before_evaluation'] is True
        assert len(list((root / 'runs').rglob('metrics.json'))) == 8
    finally:
        if root.exists():
            shutil.rmtree(root)
