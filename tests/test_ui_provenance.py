from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ui.provenance import (
    COMBINED_DIFF_FILE_NAME,
    PROVENANCE_FILE_NAME,
    STAGED_DIFF_FILE_NAME,
    UNSTAGED_DIFF_FILE_NAME,
    collect_provenance,
    load_provenance,
    write_provenance,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for provenance tests")
class ProvenanceTests(unittest.TestCase):
    def run_git(self, repo_root: Path, *args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def make_repository(self, parent: Path) -> Path:
        repo_root = parent / "repo"
        repo_root.mkdir()
        self.run_git(repo_root, "init")
        self.run_git(repo_root, "config", "user.email", "tests@example.invalid")
        self.run_git(repo_root, "config", "user.name", "FedXplore tests")
        (repo_root / "tracked.txt").write_text("base\n", encoding="utf-8")
        self.run_git(repo_root, "add", "tracked.txt")
        self.run_git(repo_root, "commit", "-m", "Initial test commit")
        return repo_root

    def test_collects_clean_repository_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = self.make_repository(Path(tmp_dir))

            provenance = collect_provenance(repo_root)

            self.assertEqual(provenance["schema_version"], 1)
            self.assertTrue(provenance["git"]["available"])
            self.assertTrue(provenance["git"]["repository"])
            self.assertEqual(provenance["git"]["repo_root"], str(repo_root.resolve()))
            self.assertFalse(provenance["git"]["dirty"])
            self.assertEqual(provenance["git"]["status_porcelain"], [])
            self.assertIsNotNone(provenance["git"]["commit"])
            self.assertIsNotNone(provenance["git"]["short_commit"])

    def test_writes_dirty_repository_diffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = self.make_repository(tmp_path)
            (repo_root / "tracked.txt").write_text("changed\n", encoding="utf-8")
            run_dir = tmp_path / "run"

            provenance = write_provenance(run_dir, repo_root)

            self.assertTrue(provenance["git"]["dirty"])
            self.assertIn("tracked.txt", provenance["git"]["modified_files"])
            self.assertIn("tracked.txt", provenance["git"]["unstaged_files"])
            self.assertEqual(load_provenance(run_dir), provenance)
            self.assertTrue((run_dir / PROVENANCE_FILE_NAME).is_file())
            self.assertIn(
                "+changed",
                (run_dir / COMBINED_DIFF_FILE_NAME).read_text(encoding="utf-8"),
            )
            self.assertIn(
                "+changed",
                (run_dir / UNSTAGED_DIFF_FILE_NAME).read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (run_dir / STAGED_DIFF_FILE_NAME).read_text(encoding="utf-8"), ""
            )

    def test_collects_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = self.make_repository(Path(tmp_dir))
            (repo_root / "new_file.txt").write_text("untracked\n", encoding="utf-8")

            provenance = collect_provenance(repo_root)

            self.assertTrue(provenance["git"]["dirty"])
            self.assertEqual(provenance["git"]["untracked_files"], ["new_file.txt"])
            self.assertEqual(provenance["git"]["modified_files"], [])

    def test_collects_staged_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = self.make_repository(Path(tmp_dir))
            (repo_root / "tracked.txt").write_text("staged\n", encoding="utf-8")
            self.run_git(repo_root, "add", "tracked.txt")

            provenance = collect_provenance(repo_root)

            self.assertTrue(provenance["git"]["dirty"])
            self.assertIn("tracked.txt", provenance["git"]["modified_files"])
            self.assertIn("tracked.txt", provenance["git"]["staged_files"])
            self.assertEqual(provenance["git"]["unstaged_files"], [])

    def test_handles_initial_repository_without_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "empty-repo"
            repo_root.mkdir()
            self.run_git(repo_root, "init")

            provenance = collect_provenance(repo_root)

            self.assertTrue(provenance["git"]["available"])
            self.assertTrue(provenance["git"]["repository"])
            self.assertIsNone(provenance["git"]["commit"])
            self.assertFalse(provenance["git"]["dirty"])

    def test_loads_legacy_run_without_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "legacy-run"
            run_dir.mkdir()

            self.assertIsNone(load_provenance(run_dir))


if __name__ == "__main__":
    unittest.main()
