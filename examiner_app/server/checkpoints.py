"""Checkpoint discovery and cached model loading for the layout-fix runs (read-only)."""
import glob
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_write_gate import load  # noqa: E402

ARMS = ('A0_control', 'A1_varied_distance', 'A2_no_block_first', 'A3_mixed_rewrites', 'A4_distance_and_rewrites')
VARIANTS = ('scalar', 'channel-wise')
SEEDS = (42, 43, 44)
LAYOUT_ROOT = ROOT / 'output' / 'layout_fix_study' / 'layout-fix-v1'


def find_checkpoint(arm, variant, seed):
    """Path to latest_checkpoint.pt for one run, or None if the run is not on disk."""
    hits = glob.glob(str(LAYOUT_ROOT / '*' / arm / f'seed_{seed}' / variant / 'latest_checkpoint.pt'))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(f'Ambiguous checkpoints for {arm}/{variant}/{seed}: {hits}')
    return Path(hits[0])


def all_runs():
    """(arm, variant, seed, checkpoint_path) for every layout-fix run present on disk, in spec order."""
    out = []
    for arm in ARMS:
        for variant in VARIANTS:
            for seed in SEEDS:
                path = find_checkpoint(arm, variant, seed)
                if path is not None:
                    out.append((arm, variant, seed, path))
    return out


def repo_relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


@lru_cache(maxsize=12)
def load_model(arm, variant, seed):
    """Eval-mode RecipeLM on CPU. Cached per (arm, variant, seed), at most 12 models resident."""
    path = find_checkpoint(arm, variant, seed)
    if path is None:
        raise FileNotFoundError(f'No checkpoint for {arm}/{variant}/seed {seed}')
    return load(path, variant, seed)
