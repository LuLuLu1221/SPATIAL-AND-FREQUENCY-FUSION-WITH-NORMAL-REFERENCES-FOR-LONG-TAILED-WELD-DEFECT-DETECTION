"""Prepare the fixed, class-conditional LVIS tail pilot for SearchDet.

SearchDet is training-free and its released code accepts one visual concept at a
time.  This script deliberately creates a reproducible localization pilot,
rather than claiming a 1,203-class detector AP: it selects every LVIS category
whose official LVIS frequency tag is ``r`` and that has at least ``min_count``
annotations in the fixed validation pilot.  One earliest image per selected
class forms the smoke run; the complete manifest remains available for a later
extension.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--negative-keywords", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--smoke-per-class", type=int, default=1)
    args = parser.parse_args()

    with open(args.annotation, encoding="utf-8") as handle:
        data = json.load(handle)
    with open(args.negative_keywords, encoding="utf-8") as handle:
        # The released SearchDet resource is a Python dictionary assignment.
        text = handle.read()
        mapping = ast.literal_eval(text[text.index("{") : text.rindex("}") + 1])

    categories = {item["id"]: item for item in data["categories"]}
    image_by_id = {item["id"]: item for item in data["images"]}
    per_category = defaultdict(list)
    for annotation in data["annotations"]:
        per_category[annotation["category_id"]].append(annotation)
    counts = Counter({cat_id: len(items) for cat_id, items in per_category.items()})

    selected_ids = sorted(
        cat_id
        for cat_id, count in counts.items()
        if categories[cat_id].get("frequency") == "r" and count >= args.min_count
    )
    selected = []
    smoke = []
    for cat_id in selected_ids:
        category = categories[cat_id]
        name = category["name"]
        if name not in mapping:
            raise KeyError(f"No official SearchDet negative keyword for {name!r}")
        annotations = sorted(per_category[cat_id], key=lambda item: (item["image_id"], item["id"]))
        record = {
            "category_id": cat_id,
            "category_name": name,
            "negative_keyword": mapping[name],
            "frequency": category["frequency"],
            "annotation_count": len(annotations),
            "image_ids": sorted({item["image_id"] for item in annotations}),
        }
        selected.append(record)
        chosen_image_ids = record["image_ids"][: args.smoke_per_class]
        for image_id in chosen_image_ids:
            targets = [item for item in annotations if item["image_id"] == image_id]
            smoke.append(
                {
                    **record,
                    "image_id": image_id,
                    "file_name": image_by_id[image_id]["file_name"],
                    "targets": [
                        {"annotation_id": item["id"], "bbox_xywh": item["bbox"], "area": item["area"]}
                        for item in targets
                    ],
                }
            )

    output = {
        "protocol": {
            "name": "SearchDet LVIS rare-class class-conditional localization pilot",
            "selection_rule": "All validation-pilot categories with official LVIS frequency='r' and >= min_count annotations.",
            "min_count": args.min_count,
            "smoke_per_class": args.smoke_per_class,
            "metric_scope": "Class-conditioned localization; not full LVIS detection AP.",
            "searchdet_supports": "Five cached positive and five cached negative web images per class, using the released negative-keyword mapping.",
        },
        "selected_categories": selected,
        "smoke_items": smoke,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"categories={len(selected)} full_instances={sum(x['annotation_count'] for x in selected)} smoke_items={len(smoke)}")
    print(path)


if __name__ == "__main__":
    main()
