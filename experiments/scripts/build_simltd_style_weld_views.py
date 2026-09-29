#!/usr/bin/env python
"""Build deterministic, train-only COCO views for a SimLTD-style weld control.

The source images and validation annotations are never modified.  D_head uses
classes with >= head_min training instances; D_tail uses nonzero classes below
that threshold; D_k selects up to k primary instances per active class.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--head-min', type=int, default=100)
    p.add_argument('--shots', type=int, default=30)
    p.add_argument('--negative-ratio', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {a.output_dir}')
    src = json.loads(a.input.read_text(encoding='utf-8'))
    images = {int(x['id']): x for x in src['images']}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    anns_by_class: dict[int, list[dict]] = defaultdict(list)
    for ann in src['annotations']:
        anns_by_image[int(ann['image_id'])].append(ann)
        anns_by_class[int(ann['category_id'])].append(ann)
    counts = {int(c['id']): len(anns_by_class[int(c['id'])]) for c in src['categories']}
    head_ids = sorted(cid for cid, n in counts.items() if n >= a.head_min)
    tail_ids = sorted(cid for cid, n in counts.items() if 0 < n < a.head_min)
    active_ids = sorted(head_ids + tail_ids)
    rng = random.Random(a.seed)
    empty_ids = sorted(set(images) - set(anns_by_image))

    def write_view(name: str, image_ids: set[int], annotations: list[dict], details: dict) -> None:
        positive_ids = {int(x['image_id']) for x in annotations}
        requested_empty = round(len(positive_ids) * a.negative_ratio)
        selected_empty = rng.sample(empty_ids, min(requested_empty, len(empty_ids)))
        selected_ids = sorted(image_ids | set(selected_empty))
        payload = {
            'info': {
                **src.get('info', {}),
                'simltd_style_weld': {
                    'view': name,
                    'source_annotation': str(a.input),
                    'seed': a.seed,
                    'head_min_instances': a.head_min,
                    'shots_per_class': a.shots,
                    'head_ids': head_ids,
                    'tail_ids': tail_ids,
                    'validation_used': False,
                    'negative_ratio': a.negative_ratio,
                    'selected_empty_images': len(selected_empty),
                    **details,
                },
            },
            'licenses': src.get('licenses', []),
            'categories': src['categories'],
            'images': [images[i] for i in selected_ids],
            'annotations': annotations,
        }
        target = a.output_dir / 'annotations' / f'instances_{name}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
        print(json.dumps({
            'view': name, 'images': len(payload['images']), 'annotations': len(annotations),
            'classes': sorted({int(x['category_id']) for x in annotations}), 'path': str(target),
        }, ensure_ascii=False))

    # Steps 1 and 2 use class-filtered COCO labels, following D_head/D_tail.
    head_anns = [x for x in src['annotations'] if int(x['category_id']) in set(head_ids)]
    write_view('stage1_head', {int(x['image_id']) for x in head_anns}, head_anns,
               {'phase': 'representation_pretraining', 'label_space': 'head_and_medium_only'})
    tail_anns = [x for x in src['annotations'] if int(x['category_id']) in set(tail_ids)]
    write_view('stage2_tail', {int(x['image_id']) for x in tail_anns}, tail_anns,
               {'phase': 'frozen_representation_tail_transfer', 'label_space': 'tail_only'})

    # Step 3 selects up to k primary instances/class.  It retains all labels on
    # selected images so the final joint fine-tuning does not turn co-occurring
    # objects into false background.  Primary support IDs document the sampling.
    primary_ids: list[int] = []
    selected_image_ids: set[int] = set()
    per_class_primary: dict[str, int] = {}
    for cid in active_ids:
        candidates = sorted(anns_by_class[cid], key=lambda x: int(x['id']))
        chosen = rng.sample(candidates, min(a.shots, len(candidates)))
        primary_ids.extend(int(x['id']) for x in chosen)
        selected_image_ids.update(int(x['image_id']) for x in chosen)
        per_class_primary[str(cid)] = len(chosen)
    joint_anns = [x for x in src['annotations'] if int(x['image_id']) in selected_image_ids]
    write_view('stage3_joint_k30', selected_image_ids, joint_anns, {
        'phase': 'head_tail_fusion_finetuning',
        'primary_support_annotation_ids': sorted(primary_ids),
        'primary_support_per_class': per_class_primary,
        'joint_labels_retained_for_selected_images': True,
    })
    summary = {
        'source_images': len(images), 'source_annotations': len(src['annotations']),
        'counts_by_class': counts, 'head_ids': head_ids, 'tail_ids': tail_ids,
        'excluded_zero_instance_ids': sorted(cid for cid, n in counts.items() if n == 0),
    }
    (a.output_dir / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

