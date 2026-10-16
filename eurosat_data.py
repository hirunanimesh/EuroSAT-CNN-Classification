"""Shared, ordered EuroSAT splits for every model in this repository."""

import csv
import json
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
EXPECTED_COUNTS = {"train": 18900, "val": 4050, "test": 4050}


def load_splits(manifest_path=ROOT / "eurosat_splits.csv",
                classes_path=ROOT / "eurosat_classes.json"):
    """Validate the supplied files without reshuffling or reassigning labels."""
    with Path(classes_path).open(encoding="utf-8") as handle:
        classes = json.load(handle)
    if (not isinstance(classes, list) or len(classes) != 10
            or any(not isinstance(name, str) or not name for name in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("Expected ten unique class names in label order")
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["relative_path", "label", "split"]:
            raise ValueError("Expected relative_path,label,split CSV columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        raw_path = row["relative_path"]
        path = PurePosixPath(raw_path)
        if (path.is_absolute() or len(path.parts) != 2
                or any(part in {".", ".."} for part in path.parts)
                or "\\" in raw_path or ":" in raw_path
                or path.as_posix() != raw_path):
            raise ValueError(f"Invalid image path: {raw_path}")
        if raw_path in seen:
            raise ValueError(f"Duplicate image across splits: {raw_path}")
        seen.add(raw_path)
        label = int(row["label"])
        if not 0 <= label < len(classes) or path.parts[0] != classes[label]:
            raise ValueError(f"Label and class folder disagree: {raw_path}")
        row["label"] = label
    counts = Counter(row["split"] for row in rows)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split counts: {dict(counts)}")
    for split in EXPECTED_COUNTS:
        if {row["label"] for row in rows if row["split"] == split} != set(range(10)):
            raise ValueError(f"Missing classes in {split}")
    return rows, classes


def validate_images(rows, root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    missing = [row["relative_path"] for row in rows
               if not (root / row["relative_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; examples: {missing[:5]}")
