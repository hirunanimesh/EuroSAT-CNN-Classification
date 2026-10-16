import csv
import json
from pathlib import Path
import tempfile
import unittest

from eurosat_data import EXPECTED_COUNTS, ROOT, load_splits, validate_images


class SharedSplitTests(unittest.TestCase):
    def test_original_manifest_order_and_labels(self):
        rows, classes = load_splits()
        self.assertEqual(len(rows), sum(EXPECTED_COUNTS.values()))
        self.assertEqual(rows[0], {"relative_path": "Residential/Residential_238.jpg",
                                   "label": 7, "split": "train"})
        self.assertEqual(classes[7], "Residential")
        self.assertEqual({row["split"] for row in rows}, set(EXPECTED_COUNTS))

    def reject_modified_row(self, change):
        with (ROOT / "eurosat_splits.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        change(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "label", "split"])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                load_splits(path)

    def test_duplicate_cannot_leak_between_splits(self):
        self.reject_modified_row(lambda rows: rows[-1].update(rows[0], split="test"))

    def test_label_must_match_folder(self):
        self.reject_modified_row(lambda rows: rows[0].update(label="0"))

    def test_counts_cannot_drift(self):
        self.reject_modified_row(lambda rows: rows[0].update(split="val"))

    def test_path_cannot_escape_dataset(self):
        self.reject_modified_row(lambda rows: rows[0].update(relative_path="../outside.jpg"))

    def test_class_mapping_cannot_be_reordered(self):
        classes = json.loads((ROOT / "eurosat_classes.json").read_text())
        classes.reverse()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.json"
            path.write_text(json.dumps(classes))
            with self.assertRaises(ValueError):
                load_splits(classes_path=path)

    def test_missing_image_reports_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Forest/Forest_1.jpg"):
                validate_images([{"relative_path": "Forest/Forest_1.jpg"}], directory)


if __name__ == "__main__":
    unittest.main()
