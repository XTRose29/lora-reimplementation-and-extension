"""Prepare or preview GLUE data for the NLU LoRA experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datasets import load_dataset

try:
    from .my_modeling import get_task_metadata
except ImportError:
    from my_modeling import get_task_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a GLUE task and export preview jsonl files.")
    parser.add_argument("--task_name", default="sst2", help="GLUE task name, e.g. sst2, mrpc, rte.")
    parser.add_argument("--output_dir", default=None, help="Defaults to data/glue_<task>.")
    parser.add_argument("--preview_samples", type=int, default=50, help="Examples to export per split.")
    parser.add_argument("--export_full", action="store_true", help="Export full splits instead of previews.")
    return parser.parse_args()


def label_to_name(dataset, label_id):
    feature = dataset.features.get("label")
    names = getattr(feature, "names", None)
    if names and label_id >= 0:
        return names[int(label_id)]
    return str(label_id)


def export_split(dataset, split_name: str, text_fields, output_path: Path, limit: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    with output_path.open("w", encoding="utf-8") as f:
        for i in range(count):
            row = dataset[i]
            record = {
                "idx": int(row.get("idx", i)),
                "split": split_name,
                "label": row.get("label"),
                "label_name": label_to_name(dataset, row.get("label", -1)),
            }
            for field in text_fields:
                record[field] = row[field]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    task_name = args.task_name.lower()
    metadata = get_task_metadata(task_name)
    text_fields = tuple(metadata["text_fields"])
    output_dir = Path(args.output_dir or f"data/glue_{task_name}")
    raw = load_dataset("glue", task_name)

    limit = None if args.export_full else args.preview_samples
    exported = {}
    for split_name, dataset in raw.items():
        suffix = split_name if args.export_full else f"{split_name}_preview"
        output_path = output_dir / f"{suffix}.jsonl"
        export_split(dataset, split_name, text_fields, output_path, limit)
        exported[split_name] = str(output_path)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task_name": task_name,
                "text_fields": text_fields,
                "splits": {name: len(ds) for name, ds in raw.items()},
                "export_full": args.export_full,
                "preview_samples": args.preview_samples,
                "exported_files": exported,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Prepared GLUE/{task_name}. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
