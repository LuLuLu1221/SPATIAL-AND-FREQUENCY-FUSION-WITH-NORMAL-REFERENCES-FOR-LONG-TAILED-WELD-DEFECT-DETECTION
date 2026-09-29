#!/usr/bin/env python
"""Run one D-FINE SimLTD-style stage with optional detector-head-only updates."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / 'third_party' / 'D-FINE'
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig, yaml_utils  # noqa: E402
from src.misc import dist_utils  # noqa: E402
from src.solver import TASKS  # noqa: E402

DEFAULT_HEAD_PATTERNS = [
    r'decoder\.(?:enc_score_head|enc_bbox_head|pre_bbox_head|dec_score_head|dec_bbox_head)',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-c', '--config', required=True)
    p.add_argument('-t', '--tuning', default=None, help='Model checkpoint to load without optimizer state.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--head-only', action='store_true', help='Freeze all but D-FINE score/regression heads.')
    p.add_argument('--print-method', default='builtin')
    p.add_argument('--print-rank', type=int, default=0)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    updates = {'seed': args.seed}
    if args.tuning:
        updates['tuning'] = args.tuning
    cfg = YAMLConfig(args.config, **updates)
    if args.tuning and 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    base = TASKS[cfg.yaml_cfg['task']]

    class StageSolver(base):
        def _setup(self):
            super()._setup()
            if not args.head_only:
                return
            model = dist_utils.de_parallel(self.model)
            patterns = [re.compile(p) for p in DEFAULT_HEAD_PATTERNS]
            trainable = []
            frozen = []
            for name, parameter in model.named_parameters():
                if any(p.search(name) for p in patterns):
                    parameter.requires_grad_(True)
                    trainable.append(name)
                else:
                    parameter.requires_grad_(False)
                    frozen.append(name)
            if not trainable:
                raise RuntimeError('No D-FINE detector-head parameters matched freeze patterns.')
            print({'simltd_style_head_only': True, 'trainable': trainable, 'frozen_count': len(frozen)}, flush=True)

    solver = StageSolver(cfg)
    solver.fit()
    dist_utils.cleanup()


if __name__ == '__main__':
    main(parse_args())

