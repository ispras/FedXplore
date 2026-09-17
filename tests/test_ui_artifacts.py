from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ui.artifacts import (
    ArtifactDownload,
    ArtifactKind,
    ArtifactMetadata,
    build_artifact_preview,
    classify_artifact,
    download_artifact,
    list_run_artifacts,
)


class FakeArtifactClient:
    def __init__(self, listings=None, payloads=None, listing_error=None) -> None:
        self.listings = listings or {}
        self.payloads = payloads or {}
        self.listing_error = listing_error
        self.list_calls: list[tuple[str, str | None]] = []
        self.download_calls: list[tuple[str, str]] = []

    def list_artifacts(self, run_id, path=None):
        self.list_calls.append((run_id, path))
        if self.listing_error:
            raise self.listing_error
        return self.listings.get(path, [])

    def download_artifacts(self, run_id, path, dst_path):
        self.download_calls.append((run_id, path))
        target = Path(dst_path) / Path(path).name
        target.write_bytes(self.payloads[path])
        return str(target)


class ArtifactListingTests(unittest.TestCase):
    def test_recursively_lists_nested_files_without_downloading(self) -> None:
        client = FakeArtifactClient(
            listings={
                None: [
                    SimpleNamespace(path="reports", is_dir=True, file_size=None),
                    SimpleNamespace(path="config.yaml", is_dir=False, file_size=24),
                ],
                "reports": [
                    SimpleNamespace(
                        path="reports/plots", is_dir=True, file_size=None
                    ),
                    SimpleNamespace(
                        path="reports/selection.csv", is_dir=False, file_size=81
                    ),
                ],
                "reports/plots": [
                    SimpleNamespace(
                        path="reports/plots/participation.png",
                        is_dir=False,
                        file_size=1500,
                    )
                ],
            }
        )

        result = list_run_artifacts(
            "run-1",
            "file:///private",
            client_factory=lambda _: client,
        )

        self.assertIsNone(result.error)
        self.assertEqual(
            result.artifacts,
            [
                ArtifactMetadata("config.yaml", 24),
                ArtifactMetadata("reports/plots/participation.png", 1500),
                ArtifactMetadata("reports/selection.csv", 81),
            ],
        )
        self.assertEqual(
            set(client.list_calls),
            {
                ("run-1", None),
                ("run-1", "reports"),
                ("run-1", "reports/plots"),
            },
        )
        self.assertEqual(client.download_calls, [])

    def test_empty_store_is_a_successful_empty_result(self) -> None:
        result = list_run_artifacts(
            "run-empty",
            client_factory=lambda _: FakeArtifactClient(),
        )

        self.assertEqual(result.artifacts, [])
        self.assertIsNone(result.error)

    def test_missing_run_and_backend_errors_are_non_fatal_and_path_safe(self) -> None:
        missing = list_run_artifacts(None)
        private_path = "/home/private/project/mlruns/123"
        failed = list_run_artifacts(
            "run-1",
            private_path,
            client_factory=lambda _: FakeArtifactClient(
                listing_error=RuntimeError(f"store failed at {private_path}")
            ),
        )

        self.assertEqual(missing.artifacts, [])
        self.assertIsNotNone(missing.error)
        self.assertEqual(failed.artifacts, [])
        self.assertIsNotNone(failed.error)
        self.assertNotIn(private_path, failed.error or "")
        self.assertNotIn("/home/", failed.error or "")

    def test_missing_mlflow_is_a_non_fatal_state(self) -> None:
        def unavailable_client(_):
            raise ImportError("mlflow is not installed")

        result = list_run_artifacts(
            "run-1",
            client_factory=unavailable_client,
        )

        self.assertEqual(result.artifacts, [])
        self.assertEqual(
            result.error,
            "MLflow is unavailable; artifacts cannot be loaded.",
        )

    def test_absolute_backend_paths_are_not_exposed_as_metadata(self) -> None:
        local_path = "/home/private/project/config.yaml"
        client = FakeArtifactClient(
            listings={
                None: [
                    SimpleNamespace(path=local_path, is_dir=False, file_size=12)
                ]
            }
        )

        result = list_run_artifacts("run-1", client_factory=lambda _: client)

        self.assertEqual(result.artifacts, [ArtifactMetadata("config.yaml", 12)])
        self.assertNotIn("/home/", repr(result.artifacts))


class ArtifactDownloadAndPreviewTests(unittest.TestCase):
    def test_viewer_reuses_unchanged_successful_download(self) -> None:
        from ui import run_ui

        artifact = ArtifactMetadata("participation_histogram.png", 128)
        loaded = ArtifactDownload(artifact, data=b"image-bytes")
        with (
            patch.object(run_ui.st, "session_state", {}),
            patch("ui.run_ui.download_artifact", return_value=loaded) as download,
        ):
            first = run_ui.load_artifact_for_viewer(
                "run-1",
                "unused-tracking-uri",
                artifact,
                key_prefix="ui_test",
            )
            second = run_ui.load_artifact_for_viewer(
                "run-1",
                "unused-tracking-uri",
                artifact,
                key_prefix="ui_test",
            )

        self.assertEqual(first.data, b"image-bytes")
        self.assertEqual(second.data, b"image-bytes")
        download.assert_called_once()

    def test_download_is_lazy_and_limited_to_selected_artifact(self) -> None:
        client = FakeArtifactClient(
            listings={
                None: [
                    SimpleNamespace(path="first.txt", is_dir=False, file_size=5),
                    SimpleNamespace(path="second.txt", is_dir=False, file_size=6),
                ]
            },
            payloads={"first.txt": b"first", "second.txt": b"second"},
        )
        result = list_run_artifacts("run-1", client_factory=lambda _: client)
        selected = result.artifacts[1]

        loaded = download_artifact(
            "run-1", None, selected, client_factory=lambda _: client
        )

        self.assertEqual(loaded.data, b"second")
        self.assertEqual(client.download_calls, [("run-1", "second.txt")])

    def test_download_failure_does_not_expose_temporary_or_tracking_path(self) -> None:
        private_path = "/home/private/mlruns"

        class FailingClient(FakeArtifactClient):
            def download_artifacts(self, run_id, path, dst_path):
                raise RuntimeError(f"failed in {dst_path} from {private_path}")

        loaded = download_artifact(
            "run-1",
            private_path,
            ArtifactMetadata("report.txt", 4),
            client_factory=lambda _: FailingClient(),
        )

        self.assertIsNone(loaded.data)
        self.assertNotIn(private_path, loaded.error or "")
        self.assertNotIn(tempfile.gettempdir(), loaded.error or "")

    def test_classifies_image_csv_text_and_binary_artifacts(self) -> None:
        self.assertIs(classify_artifact("plots/chart.PNG"), ArtifactKind.IMAGE)
        self.assertIs(classify_artifact("photo.jpeg"), ArtifactKind.IMAGE)
        self.assertIs(classify_artifact("table.CSV"), ArtifactKind.CSV)
        self.assertIs(classify_artifact("config.yaml"), ArtifactKind.TEXT)
        self.assertIs(classify_artifact("checkpoint.pt"), ArtifactKind.BINARY)

    def test_text_preview_is_bounded_by_bytes_characters_and_lines(self) -> None:
        preview = build_artifact_preview(
            ArtifactDownload(
                ArtifactMetadata("run_command.txt"),
                data=b"one\ntwo\nthree\nfour\n",
            ),
            max_bytes=18,
            max_text_characters=9,
            max_text_lines=3,
        )

        self.assertEqual(preview.text, "one\ntwo\nt")
        self.assertTrue(preview.truncated)

    def test_csv_preview_bounds_rows_columns_and_input_bytes(self) -> None:
        data = b"a,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11,12\n"
        preview = build_artifact_preview(
            ArtifactDownload(ArtifactMetadata("nested/summary.csv"), data=data),
            max_bytes=len(data),
            max_csv_rows=2,
            max_csv_columns=3,
        )

        self.assertIsNotNone(preview.dataframe)
        self.assertEqual(preview.dataframe.shape, (2, 3))
        self.assertEqual(list(preview.dataframe.columns), ["a", "b", "c"])
        self.assertTrue(preview.truncated)

        byte_bounded = build_artifact_preview(
            ArtifactDownload(ArtifactMetadata("summary.csv"), data=data),
            max_bytes=12,
            max_csv_rows=10,
            max_csv_columns=10,
        )
        self.assertTrue(byte_bounded.truncated)

    def test_large_image_is_not_returned_as_a_preview(self) -> None:
        preview = build_artifact_preview(
            ArtifactDownload(ArtifactMetadata("plot.webp"), data=b"x" * 11),
            max_bytes=10,
        )

        self.assertIs(preview.kind, ArtifactKind.IMAGE)
        self.assertTrue(preview.truncated)
        self.assertIsNotNone(preview.error)


if __name__ == "__main__":
    unittest.main()
