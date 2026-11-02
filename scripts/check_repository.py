"""Verify the release manifest, Python syntax and strict result JSONs (stdlib only)."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def reject_constant(value):
    raise ValueError('Nonstandard JSON constant: ' + value)


def main():
    count = 0
    for line in (ROOT / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Checksum mismatch: ' + name)
        count += 1
    sources = list(ROOT.rglob('*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    results = list((ROOT / 'results' / 'eval_jsons').glob('*.json'))
    if not results:
        raise ValueError('No evaluation JSONs found')
    for path in results:
        json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
    print(f'PASS: {count} checksums, {len(sources)} Python sources, {len(results)} strict evaluation JSONs')


if __name__ == '__main__':
    main()
