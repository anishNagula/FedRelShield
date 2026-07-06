import json
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset

from ultra.tasks import build_relation_graph


def resolve_dataset_root(root):
        return str(
            Path(root)
           .expanduser()
           .resolve()
        )


class FedRelShieldDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        enterprise_id,
        transform=None,
        pre_transform=build_relation_graph,
    ):
        self.enterprise_id = enterprise_id

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
        )

        self.data, self.slices = torch.load(
            self.processed_paths[0]
        )

    @property
    def raw_dir(self):
        return str(
            Path(self.root)
            / self.enterprise_id
        )

    @property
    def processed_dir(self):
        return str(
            Path(self.root)
            / self.enterprise_id
            / "processed"
        )

    @property
    def raw_file_names(self):
        return [
            "train.txt",
            "valid.txt",
            "test.txt",
            "campaigns.json",
            "provenance.jsonl",
            "statistics.json",
            "manifest.json",
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_relations(self):
        value = self[0].num_relations

        if torch.is_tensor(value):
            return int(value.item())

        return int(value)

    def download(self):
        raise RuntimeError(
            "FedRelShield enterprise artifacts must be "
            "generated locally before loading"
        )

    def process(self):
        self._verify_manifest()

        entity_vocab = {}
        relation_vocab = {}

        train_triples = self._load_train_file(
            self.raw_paths[0],
            entity_vocab,
            relation_vocab,
        )

        valid_triples = self._load_evaluation_file(
            self.raw_paths[1],
            entity_vocab,
            relation_vocab,
        )

        test_triples = self._load_evaluation_file(
            self.raw_paths[2],
            entity_vocab,
            relation_vocab,
        )

        if not train_triples:
            raise ValueError(
                "FedRelShield training split must be non-empty"
            )

        num_entities = len(entity_vocab)
        num_forward_relations = len(relation_vocab)

        train_tensor = torch.tensor(
            train_triples,
            dtype=torch.long,
        )

        valid_tensor = torch.tensor(
            valid_triples,
            dtype=torch.long,
        )

        test_tensor = torch.tensor(
            test_triples,
            dtype=torch.long,
        )

        train_target_edge_index = (
            train_tensor[:, :2].t().contiguous()
        )
        train_target_edge_type = train_tensor[:, 2]

        valid_target_edge_index = (
            valid_tensor[:, :2].t().contiguous()
        )
        valid_target_edge_type = valid_tensor[:, 2]

        test_target_edge_index = (
            test_tensor[:, :2].t().contiguous()
        )
        test_target_edge_type = test_tensor[:, 2]

        # ULTRA's transductive contract:
        # training triples define the fact graph and inverse
        # relations are added only to that graph.
        fact_edge_index = torch.cat(
            [
                train_target_edge_index,
                train_target_edge_index.flip(0),
            ],
            dim=-1,
        )

        fact_edge_type = torch.cat(
            [
                train_target_edge_type,
                train_target_edge_type
                + num_forward_relations,
            ],
            dim=0,
        )

        num_relations = 2 * num_forward_relations

        train_data = Data(
            edge_index=fact_edge_index,
            edge_type=fact_edge_type,
            num_nodes=num_entities,
            target_edge_index=train_target_edge_index,
            target_edge_type=train_target_edge_type,
            num_relations=num_relations,
        )

        valid_data = Data(
            edge_index=fact_edge_index,
            edge_type=fact_edge_type,
            num_nodes=num_entities,
            target_edge_index=valid_target_edge_index,
            target_edge_type=valid_target_edge_type,
            num_relations=num_relations,
        )

        test_data = Data(
            edge_index=fact_edge_index,
            edge_type=fact_edge_type,
            num_nodes=num_entities,
            target_edge_index=test_target_edge_index,
            target_edge_type=test_target_edge_type,
            num_relations=num_relations,
        )

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(
            self.collate(
                [
                    train_data,
                    valid_data,
                    test_data,
                ]
            ),
            self.processed_paths[0],
        )

    def _load_train_file(
        self,
        path,
        entity_vocab,
        relation_vocab,
    ):
        triples = []

        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                head, relation, tail = self._parse_line(
                    line,
                    path,
                    line_number,
                )

                head_id = self._get_or_create_id(
                    entity_vocab,
                    head,
                )

                tail_id = self._get_or_create_id(
                    entity_vocab,
                    tail,
                )

                relation_id = self._get_or_create_id(
                    relation_vocab,
                    relation,
                )

                triples.append(
                    (
                        head_id,
                        tail_id,
                        relation_id,
                    )
                )

        return triples

    def _load_evaluation_file(
        self,
        path,
        entity_vocab,
        relation_vocab,
    ):
        triples = []

        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                head, relation, tail = self._parse_line(
                    line,
                    path,
                    line_number,
                )

                if relation not in relation_vocab:
                    raise ValueError(
                        "Evaluation relation missing from "
                        "training relation vocabulary: "
                        f"{relation}"
                    )

                # Transductive evaluation requires one shared
                # entity vocabulary across all splits. Evaluation
                # entities may be registered here if absent from
                # training, although the current benchmark exporter
                # should normally prevent that situation.
                head_id = self._get_or_create_id(
                    entity_vocab,
                    head,
                )

                tail_id = self._get_or_create_id(
                    entity_vocab,
                    tail,
                )

                triples.append(
                    (
                        head_id,
                        tail_id,
                        relation_vocab[relation],
                    )
                )

        return triples

    def _parse_line(
        self,
        line,
        path,
        line_number,
    ):
        stripped = line.strip()

        if not stripped:
            raise ValueError(
                f"Empty triple at {path}:{line_number}"
            )

        fields = stripped.split("\t")

        if len(fields) != 3:
            raise ValueError(
                "Expected tab-separated triple at "
                f"{path}:{line_number}"
            )

        head, relation, tail = fields

        if not head or not relation or not tail:
            raise ValueError(
                "Triple fields must be non-empty at "
                f"{path}:{line_number}"
            )

        return head, relation, tail

    def _get_or_create_id(
        self,
        vocabulary,
        token,
    ):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

        return vocabulary[token]

    def _verify_manifest(self):
        manifest_path = (
            Path(self.raw_dir)
            / "manifest.json"
        )

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if manifest.get("enterprise_id") != self.enterprise_id:
            raise ValueError(
                "Manifest enterprise ID does not match "
                f"requested enterprise: {self.enterprise_id}"
            )

        expected_counts = manifest.get("counts")

        if not isinstance(expected_counts, dict):
            raise ValueError(
                "Manifest counts field must be a dictionary"
            )

        actual_counts = {
            "train_triples": self._count_lines(
                Path(self.raw_dir) / "train.txt"
            ),
            "valid_triples": self._count_lines(
                Path(self.raw_dir) / "valid.txt"
            ),
            "test_triples": self._count_lines(
                Path(self.raw_dir) / "test.txt"
            ),
        }

        for key, actual_count in actual_counts.items():
            if expected_counts.get(key) != actual_count:
                raise ValueError(
                    "Manifest count mismatch for "
                    f"{key}: expected "
                    f"{expected_counts.get(key)}, "
                    f"received {actual_count}"
                )

    def _count_lines(self, path):
        with open(path, "r", encoding="utf-8") as file:
            return sum(1 for _ in file)
