"""Where do block-first errors come from? Regenerates the deterministic confirmation data and maps each
wrong latest-target answer to its memory slot and its distance from the query marker.
Excludes the two default seed-43 models (no shortcut). Usage: python research_notes/analyze_block_layout_errors_20261117.py"""
import csv
import gzip
import sys
from collections import Counter
sys.path.insert(0, '.')
from confirm_block_layout import cells, sample_layouts, pattern_of

rows = list(csv.DictReader(gzip.open('research_results/block_layout_confirmation_20261117/predictions.csv.gz', 'rt')))
index = {c[0]: (i, c) for i, c in enumerate(cells())}
for seed in (42, 43, 44):
    for name in ('main_r3', 'distance12_r1', 'distance12_r3', 'distance16_r1', 'distance16_r3'):
        i, (_, keys, distance, rewrites, _) = index[name]
        records = sample_layouts(seed, i, keys, distance, rewrites, ['block_first'], 256)[0]['block_first']
        counts = Counter()
        for r in rows:
            if (r['seed'] != str(seed) or r['cell'] != name or r['layout'] != 'block_first' or r['outcome'] == 'correct'
                    or (r['initialization'], r['seed']) == ('default', '43')):
                continue
            record = records[int(r['example'])]
            assert pattern_of(record) == r['pattern']
            tokens, answer = record.tokens.tolist(), int(r['answer'])
            marker = next(a.label_position for a in record.answers if a.kind == 'latest') - 1
            slots = record.metadata['all_write_positions']
            hits = [p for p in slots if tokens[p] == answer]
            counts[f'slot {slots.index(hits[0])} ({record.metadata["write_values"].count(answer) and "target" or "unrelated"}), '
                   f'{marker - hits[0]} tokens before query' if hits else 'not in memory'] += 1
        print(seed, name, sum(counts.values()), counts.most_common(4))
