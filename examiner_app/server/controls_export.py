"""Regenerate server/data/controls_inside.json: where answers come from, from the real local checkpoints.

Run from examiner_app/server with the repo venv:

    ../../.venv/Scripts/python.exe controls_export.py

Zoology: every local model from the saved layout test, on the standard layout (shift_0) and one blank token before
the table (shift_1), with 512 examples and example seed = the model's seed. This is exactly the saved test's data, so
each accuracy is checked against cell_results.csv (matches_saved). Shortcut: the 12 local overwrite models on the
block-first layout with three rewrites, using the same settings as the shortcut page's error map (64 examples,
example seed 3). Read-only with respect to research files; writes only server/data/controls_inside.json.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SERVER = Path(__file__).resolve().parent
for entry in (SERVER.parents[1], SERVER):   # repo root (research modules), then server modules
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import confirm_block_layout as cbl  # noqa: E402
import controls_data  # noqa: E402
import shortcut  # noqa: E402

cbl.SEED_OFFSET = 3_000_000   # as app.py sets it for the shortcut evaluations
OUT = SERVER / 'data' / 'controls_inside.json'
ZOOLOGY_LAYOUTS = ('shift_0', 'shift_1')
ZOOLOGY_EXAMPLES = 512
SHORTCUT_SETTINGS = dict(layout='block_first', rewrites=3, distance=8, examples=64, example_seed=3, patch='none')


def saved_accuracy():
    rows = controls_data._csv(controls_data.ZOOLOGY_DIR / 'cell_results.csv')
    init_label = controls_data.INIT_LABEL
    return {(init_label[r['initialization']], r['variant'], int(r['seed']), r['cell']): float(r['accuracy']) for r in rows}


def main():
    saved = saved_accuracy()
    zoology = []
    for m in controls_data.zoology_models():
        if not m['local']:
            continue
        for layout in ZOOLOGY_LAYOUTS:
            r = controls_data.evaluate(m['init'], m['variant'], m['seed'], layout, ZOOLOGY_EXAMPLES, m['seed'],
                                       show_examples=0)
            ref = saved[(m['init'], m['variant'], m['seed'], layout)]
            zoology.append(dict(
                init=m['init'], variant=m['variant'], seed=m['seed'], group=m['group'], layout=layout,
                n_answers=r['n_answers'], accuracy=r['accuracy'], saved_accuracy=ref,
                matches_saved=abs(r['accuracy'] - ref) < 1e-9,
                answer_sources=r['answer_sources'], errors=r['errors'], slot_histogram=r['slot_histogram']))
            print(f"zoology {m['init']}/{m['variant']}/{m['seed']} {layout}: {r['accuracy']:.4f} "
                  f"(saved {ref:.4f}, matches={zoology[-1]['matches_saved']})", flush=True)

    shortcut_rows = []
    for m in shortcut.models():
        r = shortcut.evaluate(m['init'], m['variant'], m['seed'], show_examples=1, **SHORTCUT_SETTINGS)
        shortcut_rows.append(dict(
            init=m['init'], variant=m['variant'], seed=m['seed'], layout=SHORTCUT_SETTINGS['layout'],
            rewrites=SHORTCUT_SETTINGS['rewrites'], n_answers=r['n_examples'],
            accuracy_latest=r['accuracy_latest'], first_rewrite_rate=r['first_rewrite_rate'],
            answer_sources=r['answer_sources'], pick_histogram=r['pick_histogram'],
            slot_histogram=r['slot_histogram']))
        print(f"shortcut {m['init']}/{m['variant']}/{m['seed']}: latest {r['accuracy_latest']:.4f}", flush=True)

    data = dict(
        generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        zoology_settings=dict(examples=ZOOLOGY_EXAMPLES, example_seed='model seed', layouts=list(ZOOLOGY_LAYOUTS)),
        shortcut_settings=SHORTCUT_SETTINGS,
        zoology=zoology,
        shortcut=shortcut_rows,
        all_zoology_match_saved=all(row['matches_saved'] for row in zoology),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1), encoding='utf-8')
    print(f'wrote {OUT} (zoology rows {len(zoology)}, shortcut rows {len(shortcut_rows)}, '
          f'all match saved: {data["all_zoology_match_saved"]})', flush=True)


if __name__ == '__main__':
    main()
