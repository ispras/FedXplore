from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.dataset_utils import update_data_sources


class FrameworkPathTests(unittest.TestCase):
    def test_update_data_sources_preserves_absolute_paths(self) -> None:
        data_sources = {
            "train_map_file": ["/tmp/train_map_file.csv"],
            "test_map_file": ["/tmp/test_map_file.csv"],
        }

        updated = update_data_sources("data", data_sources)

        self.assertEqual(updated["train_map_file"], ["/tmp/train_map_file.csv"])
        self.assertEqual(updated["test_map_file"], ["/tmp/test_map_file.csv"])

    def test_update_data_sources_resolves_relative_paths_against_base_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "datasets"
            expected = (base_dir / "maps/train.csv").resolve()
            data_sources = {"train_map_file": ["maps/train.csv"]}

            updated = update_data_sources(str(base_dir), data_sources)

            self.assertEqual(updated["train_map_file"], [str(expected)])


if __name__ == "__main__":
    unittest.main()
