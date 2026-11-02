"""Seed a new training run without changing the archived training source."""
import argparse
import json
from pathlib import Path
import random
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.training_args
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for flag in ('--log_path', '--checkpoints'):
        if flag in forwarded:
            Path(forwarded[forwarded.index(flag) + 1]).parent.mkdir(parents=True, exist_ok=True)
    print('NEW_RUN:::' + json.dumps({'seed': args.seed, 'torch': torch.__version__,
                                    'numpy': np.__version__, 'arguments': forwarded}), flush=True)
    entry = Path(__file__).resolve().parents[1] / 'corrected' / 'main_nowcast_rainnet.py'
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry)] + forwarded
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
