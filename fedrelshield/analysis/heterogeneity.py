import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path


ENTITY_PREFIX_TO_TYPE = {
    "user": "User",
    "host": "Host",
    "server": "Server",
    "domaincontroller": "DomainController",
    "process": "Process",
    "file": "File",
    "service": "Service",
}


class HeterogeneityAnalyzer:
    FORMAT_VERSION = "1.0"

    def __init__(
        self,
        root,
        enterprise_ids,
    ):
        self.root = Path(root)
        self.enterprise_ids = tuple(
            sorted(enterprise_ids)
        )

        if len(self.enterprise_ids) < 2:
            raise ValueError(
                "At least two enterprises are required"
            )

        if len(self.enterprise_ids) != len(
            set(self.enterprise_ids)
        ):
            raise ValueError(
                "Duplicate enterprise IDs"
            )

    def analyze(self):
        enterprise_metrics = {
            enterprise_id:
                self._analyze_enterprise(enterprise_id)
            for enterprise_id in self.enterprise_ids
        }

        pairwise_metrics = {}

        for enterprise_a, enterprise_b in combinations(
            self.enterprise_ids,
            2,
        ):
            pair_key = (
                f"{enterprise_a}__{enterprise_b}"
            )

            pairwise_metrics[pair_key] = (
                self._analyze_pair(
                    enterprise_metrics[enterprise_a],
                    enterprise_metrics[enterprise_b],
                )
            )

        summary = self._build_summary(
            enterprise_metrics,
            pairwise_metrics,
        )

        return {
            "format_version": self.FORMAT_VERSION,
            "enterprise_ids": list(
                self.enterprise_ids
            ),
            "enterprise_metrics": enterprise_metrics,
            "pairwise_metrics": pairwise_metrics,
            "summary": summary,
        }

    def _analyze_enterprise(
        self,
        enterprise_id,
    ):
        enterprise_dir = (
            self.root / enterprise_id
        )

        self._validate_artifact_files(
            enterprise_dir
        )

        manifest = self._load_json(
            enterprise_dir / "manifest.json"
        )

        if manifest.get("enterprise_id") != enterprise_id:
            raise ValueError(
                "Manifest enterprise ID mismatch for "
                f"{enterprise_id}"
            )

        triples = self._load_all_triples(
            enterprise_dir
        )

        provenance = self._load_jsonl(
            enterprise_dir / "provenance.jsonl"
        )

        if not triples:
            raise ValueError(
                f"Enterprise has no triples: {enterprise_id}"
            )

        entities = {
            entity
            for head, _, tail in triples
            for entity in (head, tail)
        }

        relations = {
            relation
            for _, relation, _ in triples
        }

        entity_type_counts = Counter(
            self._entity_type(entity)
            for entity in entities
        )

        relation_counts = Counter(
            relation
            for _, relation, _ in triples
        )

        degree_counts = Counter()

        for head, _, tail in triples:
            degree_counts[head] += 1
            degree_counts[tail] += 1

        degree_histogram = Counter(
            degree_counts[entity]
            for entity in entities
        )

        attack_triples = {
            tuple(record["triple"])
            for record in provenance
            if record.get("attack_occurrences")
        }

        attack_occurrence_count = sum(
            len(record.get("attack_occurrences", []))
            for record in provenance
        )

        attack_triple_count = len(
            attack_triples & triples
        )

        num_entities = len(entities)
        num_triples = len(triples)

        # Directed multi-relational graph density:
        #
        # |E| / (|R| * |V| * (|V| - 1))
        #
        # Self-loops are excluded from the possible-edge count.
        possible_edges = (
            len(relations)
            * num_entities
            * (num_entities - 1)
        )

        graph_density = (
            num_triples / possible_edges
            if possible_edges > 0
            else 0.0
        )

        return {
            "enterprise_id": enterprise_id,
            "profile": manifest[
                "generation"
            ]["profile"],

            "num_entities": num_entities,
            "num_triples": num_triples,
            "num_relations": len(relations),

            "train_triples": manifest[
                "counts"
            ]["train_triples"],
            "valid_triples": manifest[
                "counts"
            ]["valid_triples"],
            "test_triples": manifest[
                "counts"
            ]["test_triples"],

            "graph_density": graph_density,

            # Fraction of unique exported triples that have
            # at least one attack occurrence.
            "attack_triple_prevalence": (
                attack_triple_count / num_triples
            ),

            # Raw attack occurrences divided by unique triples.
            # This captures repeated attack activity.
            "attack_occurrence_rate": (
                attack_occurrence_count / num_triples
            ),

            "attack_triple_count": attack_triple_count,
            "attack_occurrence_count":
                attack_occurrence_count,

            "entity_type_counts": dict(
                sorted(entity_type_counts.items())
            ),
            "entity_type_distribution":
                self._normalize_counter(
                    entity_type_counts
                ),

            "relation_counts": dict(
                sorted(relation_counts.items())
            ),
            "relation_distribution":
                self._normalize_counter(
                    relation_counts
                ),

            "degree_histogram": {
                str(degree): count
                for degree, count in sorted(
                    degree_histogram.items()
                )
            },
            "degree_distribution":
                self._normalize_degree_histogram(
                    degree_histogram
                ),

            "mean_degree": (
                sum(degree_counts.values())
                / num_entities
            ),
            "max_degree": max(
                degree_counts.values()
            ),
        }

    def _analyze_pair(
        self,
        metrics_a,
        metrics_b,
    ):
        relation_jsd = self._js_divergence(
            metrics_a["relation_distribution"],
            metrics_b["relation_distribution"],
        )

        entity_type_jsd = self._js_divergence(
            metrics_a[
                "entity_type_distribution"
            ],
            metrics_b[
                "entity_type_distribution"
            ],
        )

        degree_jsd = self._js_divergence(
            metrics_a["degree_distribution"],
            metrics_b["degree_distribution"],
        )

        density_difference = abs(
            metrics_a["graph_density"]
            - metrics_b["graph_density"]
        )

        attack_prevalence_difference = abs(
            metrics_a["attack_triple_prevalence"]
            - metrics_b["attack_triple_prevalence"]
        )

        attack_occurrence_rate_difference = abs(
            metrics_a["attack_occurrence_rate"]
            - metrics_b["attack_occurrence_rate"]
        )

        return {
            "enterprise_a":
                metrics_a["enterprise_id"],
            "enterprise_b":
                metrics_b["enterprise_id"],

            "relation_distribution_jsd":
                relation_jsd,
            "entity_type_distribution_jsd":
                entity_type_jsd,
            "degree_distribution_jsd":
                degree_jsd,

            "graph_density_difference":
                density_difference,

            "attack_prevalence_difference":
                attack_prevalence_difference,

            "attack_occurrence_rate_difference":
                attack_occurrence_rate_difference,

            # Equal-weight aggregate for convenient ordering.
            # Keep component metrics in research tables.
            "aggregate_heterogeneity_score": (
                relation_jsd
                + entity_type_jsd
                + degree_jsd
                + density_difference
                + attack_prevalence_difference
            ) / 5.0,
        }

    def _build_summary(
        self,
        enterprise_metrics,
        pairwise_metrics,
    ):
        pair_count = len(pairwise_metrics)

        if pair_count == 0:
            raise ValueError(
                "No enterprise pairs were analyzed"
            )

        metric_names = (
            "relation_distribution_jsd",
            "entity_type_distribution_jsd",
            "degree_distribution_jsd",
            "graph_density_difference",
            "attack_prevalence_difference",
            "aggregate_heterogeneity_score",
        )

        pairwise_means = {
            metric_name: (
                sum(
                    pair[metric_name]
                    for pair in pairwise_metrics.values()
                )
                / pair_count
            )
            for metric_name in metric_names
        }

        most_heterogeneous_pair = max(
            pairwise_metrics,
            key=lambda pair_key:
                pairwise_metrics[pair_key][
                    "aggregate_heterogeneity_score"
                ],
        )

        least_heterogeneous_pair = min(
            pairwise_metrics,
            key=lambda pair_key:
                pairwise_metrics[pair_key][
                    "aggregate_heterogeneity_score"
                ],
        )

        return {
            "num_enterprises":
                len(enterprise_metrics),
            "num_pairs":
                pair_count,

            "pairwise_means":
                pairwise_means,

            "most_heterogeneous_pair":
                most_heterogeneous_pair,
            "most_heterogeneous_score":
                pairwise_metrics[
                    most_heterogeneous_pair
                ][
                    "aggregate_heterogeneity_score"
                ],

            "least_heterogeneous_pair":
                least_heterogeneous_pair,
            "least_heterogeneous_score":
                pairwise_metrics[
                    least_heterogeneous_pair
                ][
                    "aggregate_heterogeneity_score"
                ],
        }

    def _load_all_triples(
        self,
        enterprise_dir,
    ):
        triples = set()

        for file_name in (
            "train.txt",
            "valid.txt",
            "test.txt",
        ):
            path = enterprise_dir / file_name

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:
                for line_number, line in enumerate(
                    file,
                    start=1,
                ):
                    stripped = line.rstrip("\n")

                    if not stripped:
                        continue

                    fields = stripped.split("\t")

                    if len(fields) != 3:
                        raise ValueError(
                            "Malformed triple at "
                            f"{path}:{line_number}"
                        )

                    triples.add(tuple(fields))

        return triples

    def _load_json(
        self,
        path,
    ):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _load_jsonl(
        self,
        path,
    ):
        records = []

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    records.append(
                        json.loads(stripped)
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Malformed JSONL record at "
                        f"{path}:{line_number}"
                    ) from error

        return records

    def _validate_artifact_files(
        self,
        enterprise_dir,
    ):
        required_files = {
            "train.txt",
            "valid.txt",
            "test.txt",
            "provenance.jsonl",
            "manifest.json",
        }

        if not enterprise_dir.is_dir():
            raise FileNotFoundError(
                "Enterprise artifact does not exist: "
                f"{enterprise_dir}"
            )

        missing_files = {
            file_name
            for file_name in required_files
            if not (
                enterprise_dir / file_name
            ).is_file()
        }

        if missing_files:
            raise FileNotFoundError(
                "Enterprise artifact missing files: "
                f"{sorted(missing_files)}"
            )

    def _entity_type(
        self,
        entity_id,
    ):
        prefix = entity_id.split("_", 1)[0]

        try:
            return ENTITY_PREFIX_TO_TYPE[prefix]
        except KeyError as error:
            raise ValueError(
                "Unknown entity ID prefix: "
                f"{entity_id}"
            ) from error

    def _normalize_counter(
        self,
        counter,
    ):
        total = sum(counter.values())

        if total <= 0:
            return {}

        return {
            key: value / total
            for key, value in sorted(
                counter.items()
            )
        }

    def _normalize_degree_histogram(
        self,
        histogram,
    ):
        total = sum(histogram.values())

        if total <= 0:
            return {}

        return {
            str(degree): count / total
            for degree, count in sorted(
                histogram.items()
            )
        }

    def _js_divergence(
        self,
        distribution_a,
        distribution_b,
    ):
        keys = sorted(
            set(distribution_a)
            | set(distribution_b)
        )

        probabilities_a = [
            distribution_a.get(key, 0.0)
            for key in keys
        ]

        probabilities_b = [
            distribution_b.get(key, 0.0)
            for key in keys
        ]

        midpoint = [
            (probability_a + probability_b) / 2.0
            for probability_a, probability_b
            in zip(
                probabilities_a,
                probabilities_b,
            )
        ]

        divergence = (
            self._kl_divergence(
                probabilities_a,
                midpoint,
            )
            + self._kl_divergence(
                probabilities_b,
                midpoint,
            )
        ) / 2.0

        # Base-2 logarithms bound JSD to [0, 1].
        return divergence

    def _kl_divergence(
        self,
        probabilities,
        reference,
    ):
        return sum(
            probability
            * math.log2(
                probability / reference_probability
            )
            for probability, reference_probability
            in zip(probabilities, reference)
            if probability > 0.0
        )
