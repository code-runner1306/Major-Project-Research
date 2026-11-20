"""Write server/data/tinystories/sample.json: a few validation-split TinyStories stories from the local parquet.

Run from examiner_app/server:  ..\\..\\.venv\\Scripts\\python.exe make_tinystories_sample.py
Uses the existing dataset_generators.generate('tinystories', ...), so nothing is downloaded. Reads only.
"""
import json
from pathlib import Path

import dataset_generators as datasets

OUT = Path(__file__).resolve().parent / 'data' / 'tinystories' / 'sample.json'
SEEDS = (0, 1, 2, 3, 4)
WINDOW = 128


def main():
    stories = []
    for seed in SEEDS:
        ex = datasets.generate('tinystories', seed, {'window': WINDOW})
        stories.append(dict(
            seed=seed,
            story_index=ex['meta']['story_index'],
            num_story_tokens=ex['meta']['num_story_tokens'],
            text=ex['text'],
            text_is_truncated=len(ex['text']) >= datasets.TINYSTORIES_TEXT_CHARS,
            tokens=ex['tokens'],
            token_text=ex['token_text'],
        ))
    payload = dict(
        dataset='roneneldan/TinyStories',
        split='validation',
        file=datasets.TINYSTORIES_PARQUET.name,
        tokenizer='GPT-2 (tiktoken)',
        window=WINDOW,
        note='text holds the first 800 characters of each story. tokens are the first window tokens, '
             'with the end-of-text token (50256) where the story ends.',
        stories=stories,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'wrote {OUT} with {len(stories)} stories')


if __name__ == '__main__':
    main()
