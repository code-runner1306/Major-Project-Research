"""Live example generators for the dataset page (GET /api/datasets, POST /api/datasets/generate).

Every generator calls the repository's own code (recipe_data, retrieval, config, layout_fix_data,
confirm_block_layout, kata_overwrite_data, zoology_layout_test, third_party/dualgoose). Nothing is copied except
the role rule for overwrite examples, which is the same rule as describe_example in legacy_prototype/export_data.py
(app.py cannot be imported here without a cycle).

Heavy modules are imported on first use; warm() imports them all in a background thread at server start. Built
examples are cached per (generator, seed, params), so repeated requests are instant.

Token conventions differ by generator (see reserved_tokens in each response): overwrite-family uses 0 = query marker
and 1 = filler; Zoology uses 0 = filler; KATA uses 0 = answer slot; Anatomy uses MASK = 63.
"""
import functools
import logging
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

VOCAB = 8192
MAX_SEED = 2 ** 32 - 1          # numpy RandomState and default_rng seed range
FILLER = 1                       # overwrite-family filler token
QUERY_MARKER = 0                 # overwrite-family query marker
MASK = 63                        # Anatomy MASK token (vocabulary 64)
LAYOUT_SEED_OFFSET = 3_000_000   # fresh layout-sampling seeds; same value app.py uses

ARM_NAMES = ('A0_control', 'A1_varied_distance', 'A2_no_block_first', 'A3_mixed_rewrites',
             'A4_distance_and_rewrites')
LAYOUT_NAMES = ('block_first', 'general_nonblock', 'block_after_one_unrelated', 'run_broken_before_latest',
                'initial_then_run')
ZOOLOGY_CELL_NAMES = ('zoology_baseline', 'shift_0', 'shift_1', 'shift_2', 'shift_4', 'shift_8', 'shift_16',
                      'near', 'far', 'memory_order', 'reverse_order', 'shift_8_far')
TINYSTORIES_DIR = Path(__file__).resolve().parent / 'data' / 'tinystories'
TINYSTORIES_PARQUET = TINYSTORIES_DIR / 'data' / 'validation-00000-of-00001-869c898b519ad725.parquet'
TINYSTORIES_TIKTOKEN_CACHE = TINYSTORIES_DIR / 'tiktoken_cache'
TINYSTORIES_WINDOWS = (64, 128, 256, 512)
TINYSTORIES_EOT = 50256          # GPT-2 <|endoftext|>, appended after every story (dataset.py)
TINYSTORIES_TEXT_CHARS = 800
TINYSTORIES_SOURCE = 'huggingface.co/datasets/roneneldan/TinyStories'


class ParamError(ValueError):
    """Unknown generator or invalid parameters (HTTP 422)."""


class Unavailable(RuntimeError):
    """Generator cannot run on this machine (HTTP 409)."""


def _int(low, high, default):
    return {'type': 'int', 'min': low, 'max': high, 'default': default}


def _choice(choices, default):
    return {'type': 'choice', 'choices': list(choices), 'default': default}


CATALOGUE = [
    dict(id='mqar', name='Multi-query associative recall (Zoology)', source='recipe_data.mqar',
         origin='port', live=True, params={}),
    dict(id='overwrite', name='Keyed overwrite (our task)',
         source='retrieval.RetrievalDataset, overwrite v4', origin='ours', live=True,
         params={'distance': _int(8, 24, 8), 'rewrites': _int(1, 4, 1)}),
    dict(id='arm', name='Layout-fix training arms A0-A4', source='layout_fix_data.build_arm_data',
         origin='ours', live=True, params={'arm': _choice(ARM_NAMES, 'A0_control')}),
    dict(id='layout', name='Named test layouts', source='confirm_block_layout.sample_layouts',
         origin='ours', live=True,
         params={'layout': _choice(LAYOUT_NAMES, 'general_nonblock'), 'distance': _int(8, 24, 8),
                 'rewrites': _int(1, 4, 1)}),
    dict(id='length', name='Longer sequences (filler inserted)', source='recipe_data.mqar with filler inserted',
         origin='ours', live=True, params={'length': _choice((64, 256, 1024), 256)}),
    dict(id='kata', name='KATA two-pass overwrite (reconstruction)', source='kata_overwrite_data.generate',
         origin='ours', live=True, params={'keys': _choice((4, 8, 16, 32), 4), 'passes': _choice((1, 2, 3, 4), 2)}),
    dict(id='anatomy_e1', name='Anatomy E1: masked recall',
         source='third_party/dualgoose toy_data.make_mqar_batch', origin='upstream', live=True,
         params={'keys': _choice((4, 8, 16, 32), 8)}),
    dict(id='anatomy_e4', name='Anatomy E4: distractor haystack',
         source='third_party/dualgoose toy_data.make_long_retrieval_batch', origin='upstream', live=True, params={}),
    dict(id='anatomy_e5', name='Anatomy E5: colliding keys',
         source='third_party/dualgoose toy_data.make_long_retrieval_batch (collision=True)', origin='upstream',
         live=True, params={}),
    dict(id='zoology_cell', name='Zoology layout test cells', source='zoology_layout_test.make_cell',
         origin='ours', live=True, params={'cell': _choice(ZOOLOGY_CELL_NAMES, 'zoology_baseline')}),
    dict(id='tinystories', name='TinyStories (language)',
         source='HF roneneldan/TinyStories, GPT-2 tokenizer, 512-token windows', origin='external', live=True,
         params={'window': _choice(TINYSTORIES_WINDOWS, 128)}),
]
SPECS = {item['id']: item for item in CATALOGUE}


def catalogue():
    return [dict(item) for item in CATALOGUE]


# ---------------------------------------------------------------- lazy imports (cached)
@functools.lru_cache(maxsize=None)
def _cbl():
    import confirm_block_layout as cbl
    cbl.SEED_OFFSET = LAYOUT_SEED_OFFSET   # idempotent: app.py sets the same value
    return cbl


@functools.lru_cache(maxsize=None)
def _layout_fix():
    import layout_fix_data
    return layout_fix_data


@functools.lru_cache(maxsize=None)
def _kata():
    import kata_overwrite_data
    return kata_overwrite_data


@functools.lru_cache(maxsize=None)
def _zoology_cells():
    import zoology_layout_test
    return zoology_layout_test


@functools.lru_cache(maxsize=None)
def _toy():
    # Append (not insert) so repository modules keep precedence over anything inside third_party.
    path = str(ROOT / 'third_party' / 'dualgoose')
    if path not in sys.path:
        sys.path.append(path)
    from dualgoose import toy_data
    return toy_data


# ---------------------------------------------------------------- parameters
def _checked_params(gid, given):
    spec = SPECS[gid]['params']
    unknown = sorted(set(given) - set(spec))
    if unknown:
        raise ParamError(f'{gid}: unknown parameter(s) {unknown}; allowed: {sorted(spec)}')
    out = {}
    for name, rule in spec.items():
        value = given.get(name, rule['default'])
        if rule['type'] == 'int':
            if not (type(value) is int and rule['min'] <= value <= rule['max']):
                raise ParamError(f"{gid}.{name} must be an integer from {rule['min']} to {rule['max']}")
        elif not any(type(value) is type(c) and value == c for c in rule['choices']):
            raise ParamError(f"{gid}.{name} must be one of {rule['choices']}")
        out[name] = value
    return out


# ---------------------------------------------------------------- shared helpers
def _scored_list(scored):
    return [dict(position=p, answer=a, kind=k) for p, (a, k) in sorted(scored.items())]


def _memory_roles(length, prefix=0, pairs=4):
    roles = ['filler'] * length
    for i in range(pairs):
        roles[prefix + 2 * i] = 'key'
        roles[prefix + 2 * i + 1] = 'value'
    return roles


def _range_text(values):
    values = tuple(values)
    return f'{values[0]} to {values[-1]}' if len(values) > 1 else str(values[0])


def _mqar_row0(seed):
    """Row 0 of a 256-example Zoology batch (mqar is not prefix-stable, so a 1-example batch would differ)."""
    from recipe_data import mqar
    tokens, labels = mqar(256, VOCAB, 64, 4, seed)
    return tokens[0].tolist(), labels[0].tolist()


MQAR_ROW_NOTE = ('Uses row 0 of recipe_data.mqar(256, 8192, 64, 4, seed). Zoology MQAR is not prefix-stable, so a '
                 'one-example batch with the same seed would give different tokens.')


# ---------------------------------------------------------------- overwrite family
def _overwrite_record(seed, distance, rewrites):
    from config import RetrievalConfig
    from retrieval import RetrievalDataset
    cfg = RetrievalConfig(vocabulary_size=VOCAB, sequence_lengths=(65,), key_counts=(4,),
                          query_distances=(distance,), overwrite_counts=(rewrites,),
                          generator_version='mqar-overwrite-v4')
    try:
        return RetrievalDataset(task='overwrite', examples=1, config=cfg, seed=seed, split='train').example(0)
    except ValueError as exc:
        raise ParamError(f'infeasible overwrite combination: {exc}')


def _arm_record(arm, seed):
    """Same draw order as layout_fix_data.build_arm_data(arm, 1, seed): first accepted draw of index 0..3."""
    from config import RetrievalConfig
    from retrieval import RetrievalDataset
    lfd = _layout_fix()
    spec = lfd.ARMS[arm]
    cfg = RetrievalConfig(vocabulary_size=VOCAB, sequence_lengths=(65,), key_counts=(lfd.KEYS,),
                          query_distances=spec['query_distances'], overwrite_counts=spec['overwrite_counts'],
                          generator_version='mqar-overwrite-v4')
    dataset = RetrievalDataset(task='overwrite', examples=4, config=cfg, seed=seed, split='train')
    for index in range(4):
        record = dataset.example(index)
        if spec['reject_block_first'] and lfd.is_block_first(record):
            continue
        return record, index
    raise ParamError(f'could not draw an example for {arm} with seed {seed}')


def _overwrite_roles(record, tokens):
    """Same rule as describe_example in legacy_prototype/export_data.py (tokens: the full 65-token example)."""
    md = record.metadata
    roles = ['filler'] * len(tokens)
    for p in md['all_write_positions']:
        if p < len(tokens):
            roles[p] = 'value'
            roles[p - 1] = 'key'
    for p in md['write_positions']:
        if p < len(tokens):
            roles[p] = 'target_value'
            roles[p - 1] = 'target_key'
    for a in record.answers:
        roles[a.label_position] = 'query_latest' if a.kind == 'latest' else 'query_unrelated'
        if a.label_position - 1 >= 0 and tokens[a.label_position - 1] == QUERY_MARKER:
            roles[a.label_position - 1] = 'marker'
    return roles


def _overwrite_view(record):
    """Answer-free model input (64 tokens): gold value after each query replaced by filler, as in to_cell."""
    full = record.tokens.tolist()
    roles = _overwrite_roles(record, full)
    tokens = list(full)
    for a in record.answers:
        if a.label_position + 1 < len(tokens):
            tokens[a.label_position + 1] = FILLER
    scored = {}
    for a in record.answers:
        if a.label_position >= 64:
            raise AssertionError('overwrite answer falls outside the 64-token model input')
        scored[a.label_position] = (int(record.labels[a.label_position]), a.kind)
    return tokens[:64], roles[:64], scored


def _overwrite_notes(record):
    md = record.metadata
    wv = list(md['write_values'])
    tk = md['target_key']
    latest = next(a for a in record.answers if a.kind == 'latest')
    other = next(a for a in record.answers if a.kind == 'unrelated')
    uk = int(record.tokens[other.label_position])
    uv = int(record.labels[other.label_position])
    if len(wv) == 2:
        rewrite = f', then rewritten with {wv[1]}'
    else:
        rewrite = f', then rewritten {len(wv) - 1} times with {", ".join(map(str, wv[1:]))} in that order'
    return [
        f'Key {tk} is written with value {wv[0]}{rewrite}.',
        f'The model is asked for key {tk}; the correct answer is {wv[-1]}, the latest write '
        f'(scored at position {latest.label_position}).',
        f'A second query asks for key {uk}; its correct answer is {uv} (scored at position {other.label_position}).',
        f'The earlier value{"s" if len(wv) > 2 else ""} {", ".join(map(str, wv[:-1]))} '
        f'{"are" if len(wv) > 2 else "is"} stale and must not be given as the answer.',
    ]


def _overwrite_meta(cbl, record, **extra):
    md = record.metadata
    return dict(pattern=cbl.pattern_of(record), write_values=list(md['write_values']), target_key=md['target_key'],
                query_distance=md['query_distance'], overwrite_count=md['overwrite_count'],
                target_first=md['target_first'], vocabulary=VOCAB, **extra)


OVERWRITE_RESERVED = {'0': 'query marker', '1': 'filler'}


def _build_overwrite(seed, p):
    cbl = _cbl()
    record = _overwrite_record(seed, p['distance'], p['rewrites'])
    tokens, roles, scored = _overwrite_view(record)
    return dict(tokens=tokens, roles=roles, scored=_scored_list(scored), notes=_overwrite_notes(record),
                meta=_overwrite_meta(cbl, record), reserved_tokens=dict(OVERWRITE_RESERVED))


def _build_arm(seed, p):
    cbl = _cbl()
    lfd = _layout_fix()
    arm = p['arm']
    record, index = _arm_record(arm, seed)
    spec = lfd.ARMS[arm]
    intro = (f'Arm {arm} draws query distances from {_range_text(spec["query_distances"])} and rewrite counts from '
             f'{_range_text(spec["overwrite_counts"])}; this is draw {index} of its random stream'
             + ('; block-first layouts are rejected in this arm.' if spec['reject_block_first'] else '.'))
    tokens, roles, scored = _overwrite_view(record)
    return dict(tokens=tokens, roles=roles, scored=_scored_list(scored),
                notes=[intro, *_overwrite_notes(record)[:4]],
                meta=_overwrite_meta(cbl, record, arm=arm, draw_index=index), reserved_tokens=dict(OVERWRITE_RESERVED))


def _build_layout(seed, p):
    cbl = _cbl()
    layout, distance, rewrites = p['layout'], p['distance'], p['rewrites']
    allowed = ('general_nonblock', *cbl.special_patterns(rewrites))
    if layout not in allowed:
        raise ParamError(f"layout '{layout}' is not valid for rewrites={rewrites}; choose one of {list(allowed)}")
    try:
        chosen, _ = cbl.sample_layouts(seed, 0, 4, distance, rewrites, [layout], 1)
    except (RuntimeError, ValueError) as exc:
        raise ParamError(f'could not sample layout {layout}: {exc}')
    record = chosen[layout][0]
    pattern = cbl.pattern_of(record)
    tokens, roles, scored = _overwrite_view(record)
    intro = (f'Layout {layout}: T marks a write of the queried key and U an unrelated write, in memory order, '
             f'so this example has the pattern {pattern}.')
    return dict(tokens=tokens, roles=roles, scored=_scored_list(scored),
                notes=[intro, *_overwrite_notes(record)[:4]],
                meta=_overwrite_meta(cbl, record, layout=layout), reserved_tokens=dict(OVERWRITE_RESERVED))


# ---------------------------------------------------------------- Zoology MQAR family
def _build_mqar(seed, p):
    tokens, labels = _mqar_row0(seed)
    roles = _memory_roles(64)
    scored = {i: (labels[i], 'recall') for i in range(64) if labels[i] != -100}
    for i in scored:
        roles[i] = 'query_key'
    first = min(scored)
    memory = {tokens[i]: tokens[i + 1] for i in range(0, 8, 2)}
    notes = [
        'Positions 0 to 7 hold four key-value pairs, one after another (keys at even positions, values at odd).',
        f'The other positions are filler token 0, except the queried keys: each memory key is asked for once later.',
        f'The first query is key {tokens[first]} at position {first}; its correct answer is {labels[first]}, '
        f'the value paired with that key in the memory.',
    ]
    return dict(tokens=tokens, roles=roles, scored=_scored_list(scored), notes=notes,
                meta=dict(note=MQAR_ROW_NOTE, pairs=4, vocabulary=VOCAB, memory=[[k, v] for k, v in memory.items()]),
                reserved_tokens={'0': 'filler / answer slot'})


def _build_length(seed, p):
    length = p['length']
    extra = length - 64
    base_tokens, base_labels = _mqar_row0(seed)
    tokens = base_tokens[:8] + [0] * extra + base_tokens[8:]
    labels = base_labels[:8] + [-100] * extra + base_labels[8:]
    roles = _memory_roles(length)
    scored = {i: (labels[i], 'recall') for i in range(length) if labels[i] != -100}
    for i in scored:
        roles[i] = 'query_key'
    first = min(scored)
    if extra:
        size_note = (f'The 8 memory tokens are unchanged and {extra} filler tokens are inserted after them, '
                     f'so the row is {length} tokens long.')
    else:
        size_note = 'No filler is inserted, so this is the 64-token row.'
    notes = [
        size_note,
        f'The queries are pushed {extra} positions later; the first asks for key {tokens[first]} at position '
        f'{first}, and its correct answer is {labels[first]}.',
    ]
    return dict(tokens=tokens, roles=roles, scored=_scored_list(scored), notes=notes,
                meta=dict(note=MQAR_ROW_NOTE, base_length=64, inserted_filler=extra, length=length,
                          vocabulary=VOCAB),
                reserved_tokens={'0': 'filler'})


ZOOLOGY_MODE_TEXT = {
    'upstream': 'the unmodified Zoology MQAR layout',
    'power': 'queries at power-law gaps, with near ones much more likely',
    'near': 'all queries in the first available slots',
    'far': 'all queries in the last available slots',
    'memory_order': 'queries asked in the same order as the memory',
    'reverse_order': 'queries asked in the reverse of the memory order',
}


def _build_zoology_cell(seed, p):
    cells = {name: (prefix, mode) for name, prefix, mode in _zoology_cells().CELLS}
    prefix, mode = cells[p['cell']]
    if mode == 'upstream':
        tokens, labels = _mqar_row0(seed)
        cell_note = MQAR_ROW_NOTE
    else:
        t, l, _, _ = _zoology_cells().make_cell(1, seed, prefix, mode)
        tokens, labels = t[0].tolist(), l[0].tolist()
        cell_note = 'Row 0 of a one-example batch from zoology_layout_test.make_cell (this cell is not upstream MQAR).'
    roles = _memory_roles(64, prefix=prefix)
    scored = {i: (labels[i], 'recall') for i in range(64) if labels[i] != -100}
    for i in scored:
        roles[i] = 'query_key'
    first = min(scored)
    where = (f'The key-value memory starts after {prefix} filler tokens' if prefix
             else 'The key-value memory starts at position 0')
    notes = [
        f'Cell {p["cell"]}: {ZOOLOGY_MODE_TEXT[mode]}.',
        f'{where}; keys and values alternate after it.',
        f'The first query is key {tokens[first]} at position {first}; its correct answer is {labels[first]}.',
    ]
    return dict(tokens=tokens, roles=roles, scored=_scored_list(scored), notes=notes,
                meta=dict(cell=p['cell'], prefix=prefix, mode=mode, note=cell_note, vocabulary=VOCAB),
                reserved_tokens={'0': 'filler'})


# ---------------------------------------------------------------- KATA
def _build_kata(seed, p):
    keys, passes = p['keys'], p['passes']
    tokens, labels, _writes = _kata().generate(1, keys, passes, seed, 'train', partial_last=False)
    toks, labs = tokens[0].tolist(), labels[0].tolist()
    written = 2 * keys * passes
    roles = []
    for i, token in enumerate(toks):
        if token == 0:
            roles.append('answer_slot')
        elif i < written:
            roles.append('key' if i % 2 == 0 else 'value')
        else:
            roles.append('query_key')
    scored = {i: (labs[i], 'latest') for i in range(len(toks)) if labs[i] != -100}
    first = min(scored)
    key = toks[first]
    writes = [toks[i + 1] for i in range(0, written, 2) if toks[i] == key]
    notes = [
        f'Each of the {keys} keys is written in each of {passes} pass{"es" if passes > 1 else ""}, in a new random '
        f'order each pass, with a new value each time.',
        f'Key {key} is written with values {", ".join(map(str, writes))}, in order; it is then asked for at '
        f'position {first}, and its correct answer is {labs[first]}, the value from its last write.',
        'The queries come after all writes; each is a key followed by an answer slot (token 0). No filler is used.',
    ]
    return dict(tokens=toks, roles=roles, scored=_scored_list(scored), notes=notes,
                meta=dict(keys=keys, passes=passes, length=len(toks), vocabulary=VOCAB),
                reserved_tokens={'0': 'answer slot'})


# ---------------------------------------------------------------- TinyStories (offline, validation split)
@functools.lru_cache(maxsize=1)
def _tinystories_texts():
    """The validation 'text' column, read once from the local parquet."""
    import pyarrow.parquet as pq
    return tuple(pq.read_table(TINYSTORIES_PARQUET, columns=['text']).column('text').to_pylist())


@functools.lru_cache(maxsize=1)
def _gpt2():
    os.environ['TIKTOKEN_CACHE_DIR'] = str(TINYSTORIES_TIKTOKEN_CACHE)   # set before tiktoken is used
    import tiktoken
    return tiktoken.get_encoding('gpt2')


def _tinystories_ready():
    """Raise Unavailable (409) naming what is missing; otherwise make sure both caches load."""
    if not TINYSTORIES_PARQUET.is_file():
        raise Unavailable(f'TinyStories validation file is missing: {TINYSTORIES_PARQUET}')
    if not TINYSTORIES_TIKTOKEN_CACHE.is_dir():
        raise Unavailable(f'GPT-2 tokenizer cache directory is missing: {TINYSTORIES_TIKTOKEN_CACHE}')
    try:
        _tinystories_texts()
    except Exception as exc:  # noqa: BLE001 - reported to the caller as 409
        raise Unavailable(f'TinyStories validation file could not be read ({TINYSTORIES_PARQUET}): {exc}')
    try:
        _gpt2()
    except Exception as exc:  # noqa: BLE001
        raise Unavailable(f'GPT-2 tokenizer could not be loaded from {TINYSTORIES_TIKTOKEN_CACHE}: {exc}')


def _build_tinystories(seed, p):
    """Story seed % 21990 of the validation split, tokenised as dataset.py does (story tokens + EOT)."""
    texts = _tinystories_texts()
    index = seed % len(texts)
    text = texts[index]
    enc = _gpt2()
    story = enc.encode(text, allowed_special='all')
    window = p['window']
    tokens = (story + [TINYSTORIES_EOT])[:window]
    roles = ['eos' if t == TINYSTORIES_EOT else 'text' for t in tokens]
    n = len(story)
    if n < window:
        reading = f'It is {n} GPT-2 tokens long, so all of it is shown, followed by the end-of-text token.'
    else:
        reading = f'It is split into {n} GPT-2 tokens; the first {window} are shown.'
    notes = [
        f'This is story {index + 1:,} of {len(texts):,} in the TinyStories validation set.',
        reading,
        'A language model reads them left to right and is graded on predicting each next token.',
    ]
    shown = text[:TINYSTORIES_TEXT_CHARS] + ('...' if len(text) > TINYSTORIES_TEXT_CHARS else '')
    return dict(tokens=tokens, roles=roles, scored=[], notes=notes,
                token_text=[enc.decode([t]) for t in tokens], text=shown,
                meta=dict(story_index=index, num_story_tokens=n, window=window, split='validation',
                          source=TINYSTORIES_SOURCE, file=TINYSTORIES_PARQUET.name,
                          tokenizer='GPT-2 (tiktoken), 50,257 tokens'),
                reserved_tokens={str(TINYSTORIES_EOT): 'end of text (EOT)'})


# ---------------------------------------------------------------- Anatomy (upstream dualgoose)
def _anatomy_scored(clean, target):
    return {p: (clean[p], 'masked_value') for p in target}


def _build_anatomy_e1(seed, p):
    import torch
    toy = _toy()
    keys = p['keys']
    length = 2 * keys + 8
    batch = toy.make_mqar_batch(1, length, 64, MASK, pairs=keys, query_layout='after_table',
                                generator=torch.Generator().manual_seed(seed))
    clean, corrupted = batch.clean[0].tolist(), batch.corrupted[0].tolist()
    target = batch.target_mask[0].nonzero().flatten().tolist()
    roles = []
    for i in range(length):
        if i < 2 * keys:
            roles.append('key' if i % 2 == 0 else 'value')
        elif i == 2 * keys:
            roles.append('query_key')
        elif i == 2 * keys + 1:
            roles.append('mask')
        else:
            roles.append('noise')
    for q in target:
        roles[q] = 'mask'
        roles[q - 1] = 'query_key'
    scored = _anatomy_scored(clean, target)
    qk = clean[2 * keys]
    answer = clean[target[0]]
    notes = [
        f'The {keys} key-value pairs sit at positions 0 to {2 * keys - 1}: keys are the tokens 5 to {4 + keys} at '
        'even positions, values at odd positions.',
        f'The query key {qk} is at position {2 * keys} and its value is hidden behind MASK (token {MASK}) at '
        f'position {2 * keys + 1}.',
        f'The correct answer is {answer}, the value paired with key {qk} in the table.',
        'The rest of the row is random noise tokens.',
    ]
    return dict(tokens=corrupted, roles=roles, scored=_scored_list(scored), notes=notes,
                meta=dict(pairs=keys, length=length, vocabulary=64, mask_token=MASK,
                          source='upstream dualgoose toy_data.make_mqar_batch (query_layout after_table)'),
                reserved_tokens={str(MASK): 'MASK (answer to fill in)'})


def _build_anatomy_long(seed, collision):
    import torch
    toy = _toy()
    batch = toy.make_long_retrieval_batch(1, 64, 64, MASK, pairs=4, n_queries=4, gap=None, collision=collision,
                                          generator=torch.Generator().manual_seed(seed))
    clean, corrupted = batch.clean[0].tolist(), batch.corrupted[0].tolist()
    target = batch.target_mask[0].nonzero().flatten().tolist()
    haystack = range(8, 56)           # 48 tokens after the 4-pair table, before the 4 queries
    roles = ['noise'] * 64
    for i in range(8):
        roles[i] = 'key' if i % 2 == 0 else 'value'
    if collision:
        for i in haystack:
            roles[i] = 'decoy_key' if i % 2 == 0 else 'decoy_value'
    for q in target:
        roles[q] = 'mask'
        roles[q - 1] = 'query_key'
    scored = _anatomy_scored(clean, target)
    first_query = target[0] - 1
    qk = clean[first_query]
    table_value = clean[2 * (qk - 5) + 1]
    if collision:
        notes = [
            'The table is four key-value pairs at positions 0 to 7 (keys 5 to 8).',
            'The 48-token haystack is 24 decoy pairs that reuse the table keys with other values.',
            f'The first query asks for key {qk}; the answer is its table binding, {table_value}, the first binding in '
            'the row.',
        ]
    else:
        notes = [
            'The table is four key-value pairs at positions 0 to 7 (keys 5 to 8).',
            'The 48-token haystack contains no key tokens, only distractors.',
            f'The first query asks for key {qk}; its table value is {table_value}, which is the answer.',
        ]
    notes.append('Each of the four queries is a table key followed by MASK (token 63).')
    return dict(tokens=corrupted, roles=roles, scored=_scored_list(scored), notes=notes,
                meta=dict(pairs=4, queries=4, gap=48, collision=collision, length=64, vocabulary=64,
                          mask_token=MASK, source='upstream dualgoose toy_data.make_long_retrieval_batch'),
                reserved_tokens={str(MASK): 'MASK (answer to fill in)'})


_BUILDERS = {
    'mqar': _build_mqar,
    'overwrite': _build_overwrite,
    'arm': _build_arm,
    'layout': _build_layout,
    'length': _build_length,
    'kata': _build_kata,
    'anatomy_e1': _build_anatomy_e1,
    'anatomy_e4': lambda seed, p: _build_anatomy_long(seed, False),
    'anatomy_e5': lambda seed, p: _build_anatomy_long(seed, True),
    'zoology_cell': _build_zoology_cell,
    'tinystories': _build_tinystories,
}


@functools.lru_cache(maxsize=512)
def _build(gid, seed, items):
    return _BUILDERS[gid](seed, dict(items))


def generate(gid, seed, params=None):
    """One live example. Raises ParamError (422) or Unavailable (409)."""
    if gid not in SPECS:
        raise ParamError(f"unknown generator '{gid}'")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ParamError(f'seed must be an integer from 0 to {MAX_SEED}')
    if not isinstance(params, dict) and params is not None:
        raise ParamError('params must be an object')
    values = _checked_params(gid, params or {})
    if not SPECS[gid]['live']:
        raise Unavailable(f'{gid} is not available')
    if gid == 'tinystories':
        _tinystories_ready()
    built = _build(gid, seed, tuple(sorted(values.items())))
    out = dict(generator=gid, tokens=list(built['tokens']), roles=list(built['roles']),
               scored=[dict(s) for s in built['scored']], notes=list(built['notes']),
               meta=dict(built['meta']), reserved_tokens=dict(built['reserved_tokens']))
    if 'token_text' in built:
        out['token_text'] = list(built['token_text'])
        out['text'] = built['text']
    return out


def warm():
    """Import every heavy module and build each live generator's default example (run in a background thread)."""
    for loader in (_cbl, _layout_fix, _kata, _zoology_cells, _toy):
        try:
            loader()
        except Exception:  # noqa: BLE001 - a failed warm-up must not stop the server
            log.exception('dataset warm-up failed in %s', loader.__name__)
    for item in CATALOGUE:
        if item['live']:
            try:
                generate(item['id'], 0, {})
            except Exception:  # noqa: BLE001
                log.exception('dataset warm-up failed for %s', item['id'])


def start_warm_thread():
    thread = threading.Thread(target=warm, name='dataset-warm', daemon=True)
    thread.start()
    return thread
