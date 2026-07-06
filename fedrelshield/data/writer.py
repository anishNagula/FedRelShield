import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable

from fedrelshield.data.exporter import ExportedDataset, Triple


class EnterpriseDatasetWriter:
    TRIPLE_SEPARATOR = "\t"
    TEXT_ENCODING = "utf-8"
    FORMAT_VERSION = "1.0"

    ARTIFACT_FILES = (
        "train.txt",
        "valid.txt",
        "test.txt",
        "campaigns.json",
        "provenance.jsonl",
        "statistics.json",
    )

    def write(
        self,
        output_dir,
        enterprise_id: str,
        dataset: ExportedDataset,
        generation_metadata,
    ) -> Dict[str, str]:
        if not enterprise_id:
            raise ValueError("enterprise_id must be non-empty")

        if not isinstance(generation_metadata, dict):
            raise ValueError(
                "generation_metadata must be a dictionary"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.write_triples(
            output_dir=output_path,
            dataset=dataset,
        )

        self.write_metadata(
            output_dir=output_path,
            dataset=dataset,
        )

        file_hashes = self._compute_artifact_hashes(
            output_path
        )

        manifest = self._build_manifest(
        enterprise_id=enterprise_id,
            dataset=dataset,
            file_hashes=file_hashes,
            generation_metadata=generation_metadata,
        )

        self._atomic_write_text(
            path=output_path / "manifest.json",
            content=self._serialize_json(manifest),
        )

        self.verify_artifact(output_path)

        return {
            **file_hashes,
            "manifest.json": self.compute_sha256(
                output_path / "manifest.json"
            ),
        }

    def write_triples(
        self,
        output_dir,
        dataset: ExportedDataset,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._write_triple_file(
            output_path / "train.txt",
            dataset.train_triples,
        )

        self._write_triple_file(
            output_path / "valid.txt",
            dataset.valid_triples,
        )

        self._write_triple_file(
            output_path / "test.txt",
            dataset.test_triples,
        )

    def write_metadata(
        self,
        output_dir,
        dataset: ExportedDataset,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._write_campaigns(
            output_path / "campaigns.json",
            dataset,
        )

        self._write_provenance(
            output_path / "provenance.jsonl",
            dataset,
        )

        self._write_statistics(
            output_path / "statistics.json",
            dataset,
        )

    def verify_artifact(
        self,
        output_dir,
    ):
        output_path = Path(output_dir)

        expected_files = (
            set(self.ARTIFACT_FILES)
            | {"manifest.json"}
        )

        actual_files = {
            path.name
            for path in output_path.iterdir()
            if path.is_file()
        }

        missing_files = expected_files - actual_files

        if missing_files:
            raise ValueError(
                "Enterprise artifact is missing files: "
                f"{sorted(missing_files)}"
            )

        manifest_path = output_path / "manifest.json"

        with open(
            manifest_path,
            "r",
            encoding=self.TEXT_ENCODING,
        ) as file:
            manifest = json.load(file)

        if manifest.get("format_version") != self.FORMAT_VERSION:
            raise ValueError(
                "Unsupported artifact format version: "
                f"{manifest.get('format_version')}"
            )

        manifest_hashes = manifest.get("files")

        if not isinstance(manifest_hashes, dict):
            raise ValueError(
                "Manifest files field must be a dictionary"
            )

        if set(manifest_hashes) != set(self.ARTIFACT_FILES):
            raise ValueError(
                "Manifest file set does not match artifact contract"
            )

        for file_name in self.ARTIFACT_FILES:
            expected_hash = manifest_hashes[file_name]

            actual_hash = self.compute_sha256(
                output_path / file_name
            )

            if expected_hash != actual_hash:
                raise ValueError(
                    "Artifact hash mismatch for "
                    f"{file_name}: expected {expected_hash}, "
                    f"received {actual_hash}"
                )

    def _write_triple_file(
        self,
        path: Path,
        triples: Iterable[Triple],
    ):
        serialized = self._serialize_triples(triples)

        self._atomic_write_text(
            path=path,
            content=serialized,
        )

    def _write_campaigns(
        self,
        path: Path,
        dataset: ExportedDataset,
    ):
        campaign_data = {
            "campaigns": [
                asdict(campaign)
                for campaign in dataset.campaigns
            ],
            "splits": {
                "train_campaign_ids": list(
                    dataset.campaign_split.train_campaign_ids
                ),
                "valid_campaign_ids": list(
                    dataset.campaign_split.valid_campaign_ids
                ),
                "test_campaign_ids": list(
                    dataset.campaign_split.test_campaign_ids
                ),
            },
        }

        self._atomic_write_text(
            path=path,
            content=self._serialize_json(campaign_data),
        )

    def _write_provenance(
        self,
        path: Path,
        dataset: ExportedDataset,
    ):
        lines = [
            json.dumps(
                asdict(record),
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in dataset.provenance
        ]

        content = (
            "\n".join(lines) + "\n"
            if lines
            else ""
        )

        self._atomic_write_text(
            path=path,
            content=content,
        )

    def _write_statistics(
        self,
        path: Path,
        dataset: ExportedDataset,
    ):
        self._atomic_write_text(
            path=path,
            content=self._serialize_json(
                asdict(dataset.statistics)
            ),
        )

    def _build_manifest(
        self,
        enterprise_id: str,
        dataset: ExportedDataset,
        file_hashes: Dict[str, str],
        generation_metadata,
    ):
        return {
            "format_version": self.FORMAT_VERSION,
            "enterprise_id": enterprise_id,
            "generation": generation_metadata,
            "counts": {
                "train_triples": len(dataset.train_triples),
                "valid_triples": len(dataset.valid_triples),
                "test_triples": len(dataset.test_triples),
                "campaigns": len(dataset.campaigns),
                "provenance_records": len(dataset.provenance),
            },
            "files": {
                file_name: file_hashes[file_name]
                for file_name in self.ARTIFACT_FILES
            },
        }


    def _compute_artifact_hashes(
        self,
        output_path: Path,
    ) -> Dict[str, str]:
        return {
            file_name: self.compute_sha256(
                output_path / file_name
            )
            for file_name in self.ARTIFACT_FILES
        }

    def _serialize_triples(
        self,
        triples: Iterable[Triple],
    ) -> str:
        ordered_triples = sorted(triples)

        lines = [
            self.TRIPLE_SEPARATOR.join(triple)
            for triple in ordered_triples
        ]

        if not lines:
            return ""

        return "\n".join(lines) + "\n"

    def _serialize_json(
        self,
        value,
    ) -> str:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    def _atomic_write_text(
        self,
        path: Path,
        content: str,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )

        try:
            with os.fdopen(
                file_descriptor,
                "w",
                encoding=self.TEXT_ENCODING,
                newline="\n",
            ) as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())

            os.replace(
                temporary_path,
                path,
            )

        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

            raise

    def compute_sha256(
        self,
        path,
    ) -> str:
        digest = hashlib.sha256()

        with open(path, "rb") as file:
            while True:
                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()
