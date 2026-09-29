#!/usr/bin/env python
"""Fuse SimLTD-style D-FINE head/tail classifier rows by class ID.

D-FINE regression heads are class agnostic, so their stage-1 parameters are
preserved.  Only class-dependent score heads and denoising class embeddings use
tail rows from the transfer model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head-checkpoint', type=Path, required=True)
    p.add_argument('--tail-checkpoint', type=Path, required=True)
    p.add_argument('--tail-ids', required=True, help='Comma-separated zero-based category IDs.')
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args()


def model_state(checkpoint: dict) -> dict:
    return checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']


def main() -> None:
    a = parse_args()
    if a.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {a.output}')
    head_ckpt = torch.load(a.head_checkpoint, map_location='cpu')
    tail_ckpt = torch.load(a.tail_checkpoint, map_location='cpu')
    head = model_state(head_ckpt)
    tail = model_state(tail_ckpt)
    merged = {k: v.clone() if torch.is_tensor(v) else v for k, v in head.items()}
    tail_ids = [int(x) for x in a.tail_ids.split(',') if x]
    eligible = [
        key for key in merged
        if key in tail and (
            key.startswith('decoder.enc_score_head.') or
            key.startswith('decoder.dec_score_head.') or
            key == 'decoder.denoising_class_embed.weight'
        )
    ]
    changed = []
    for key in eligible:
        if merged[key].shape != tail[key].shape:
            continue
        for cid in tail_ids:
            if cid < merged[key].shape[0]:
                merged[key][cid].copy_(tail[key][cid])
        changed.append(key)
    payload = {
        'model': merged,
        'simltd_style_fusion': {
            'head_checkpoint': str(a.head_checkpoint),
            'tail_checkpoint': str(a.tail_checkpoint),
            'tail_ids': tail_ids,
            'fused_parameters': changed,
            'bbox_head_policy': 'preserve_stage1_class_agnostic_regression_heads',
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, a.output)
    print(payload['simltd_style_fusion'], flush=True)


if __name__ == '__main__':
    main()
