from collections import OrderedDict

import torch


class RelationAwareAggregator:
    def __init__(
        self,
        relation_distributions,
        divergence_strength=1.0,
        alpha=1.0,
    ):
        if divergence_strength < 0:
            raise ValueError(
                "divergence_strength must be non-negative"
            )

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be in [0, 1]"
            )

        if not relation_distributions:
            raise ValueError(
                "relation_distributions must be non-empty"
            )

        self.divergence_strength = float(
            divergence_strength
        )

        self.alpha = float(alpha)

        relation_names = sorted({
            relation
            for distribution
            in relation_distributions.values()
            for relation in distribution
        })

        if not relation_names:
            raise ValueError(
                "At least one relation is required"
            )

        self.relation_names = tuple(relation_names)

        self.relation_distributions = {}

        for client_id, distribution in (
            relation_distributions.items()
        ):
            if not client_id:
                raise ValueError(
                    "client IDs must be non-empty"
                )

            if any(
                value < 0
                for value in distribution.values()
            ):
                raise ValueError(
                    "Relation distributions cannot "
                    "contain negative values"
                )

            total = sum(
                distribution.values()
            )

            if total <= 0:
                raise ValueError(
                    "Relation distributions must have "
                    "positive total mass"
                )

            self.relation_distributions[
                client_id
            ] = {
                relation:
                    float(
                        distribution.get(
                            relation,
                            0.0,
                        )
                    ) / total
                for relation in self.relation_names
            }

    @staticmethod
    def _js_divergence(
        distribution_a,
        distribution_b,
    ):
        midpoint = {
            relation: (
                distribution_a[relation]
                + distribution_b[relation]
            ) / 2.0
            for relation in distribution_a
        }

        def kl_divergence(
            distribution,
            reference,
        ):
            value = 0.0

            for relation, probability in (
                distribution.items()
            ):
                if probability > 0.0:
                    value += (
                        probability
                        * torch.log(
                            torch.tensor(
                                probability
                                / reference[relation],
                                dtype=torch.float64,
                            )
                        ).item()
                    )

            return value

        return 0.5 * (
            kl_divergence(
                distribution_a,
                midpoint,
            )
            + kl_divergence(
                distribution_b,
                midpoint,
            )
        )

    def _build_global_distribution(
        self,
        client_results,
    ):
        relation_counts = {
            relation: 0.0
            for relation in self.relation_names
        }

        total_examples = sum(
            result.num_examples
            for result in client_results
        )

        if total_examples <= 0:
            raise ValueError(
                "Positive total example count required"
            )

        for result in client_results:
            distribution = (
                self.relation_distributions[
                    result.client_id
                ]
            )

            for relation in self.relation_names:
                relation_counts[relation] += (
                    result.num_examples
                    * distribution[relation]
                )

        return {
            relation:
                relation_counts[relation]
                / total_examples
            for relation in self.relation_names
        }

    def fedavg_weights(
        self,
        client_results,
    ):
        client_results = tuple(client_results)

        total_examples = sum(
            result.num_examples
            for result in client_results
        )

        if total_examples <= 0:
            raise ValueError(
                "Positive total example count required"
            )

        return {
            result.client_id:
                result.num_examples / total_examples
            for result in client_results
        }

    def relation_aware_weights(
        self,
        client_results,
    ):
        client_results = tuple(client_results)

        global_distribution = (
            self._build_global_distribution(
                client_results
            )
        )

        unnormalized_weights = {}

        for result in client_results:
            divergence = self._js_divergence(
                self.relation_distributions[
                    result.client_id
                ],
                global_distribution,
            )

            unnormalized_weights[
                result.client_id
            ] = (
                result.num_examples
                * torch.exp(
                    torch.tensor(
                        -self.divergence_strength
                        * divergence,
                        dtype=torch.float64,
                    )
                ).item()
            )

        normalizer = sum(
            unnormalized_weights.values()
        )

        if normalizer <= 0:
            raise RuntimeError(
                "Aggregation weight normalizer "
                "must be positive"
            )

        return {
            client_id:
                weight / normalizer
            for client_id, weight
            in unnormalized_weights.items()
        }

    def aggregation_weights(
        self,
        client_results,
    ):
        client_results = tuple(client_results)

        if not client_results:
            raise ValueError(
                "At least one client result is required"
            )

        result_ids = tuple(
            result.client_id
            for result in client_results
        )

        if len(result_ids) != len(set(result_ids)):
            raise ValueError(
                "Client result IDs must be unique"
            )

        if set(result_ids) != set(
            self.relation_distributions
        ):
            raise ValueError(
                "Client results do not match configured "
                "relation distributions"
            )

        for result in client_results:
            if result.num_examples <= 0:
                raise ValueError(
                    "Client example count must be positive"
                )

        fedavg_weights = self.fedavg_weights(
            client_results
        )

        relation_aware_weights = (
            self.relation_aware_weights(
                client_results
            )
        )

        return {
            client_id: (
                (1.0 - self.alpha)
                * fedavg_weights[client_id]
                + self.alpha
                * relation_aware_weights[client_id]
            )
            for client_id in result_ids
        }

    def aggregate(
        self,
        client_results,
    ):
        client_results = tuple(client_results)

        weights = self.aggregation_weights(
            client_results
        )

        reference_state = client_results[0].state

        reference_keys = tuple(
            reference_state.keys()
        )

        for result in client_results:
            if (
                tuple(result.state.keys())
                != reference_keys
            ):
                raise ValueError(
                    "Client model-state keys do not match"
                )

            for name in reference_keys:
                if (
                    result.state[name].shape
                    != reference_state[name].shape
                ):
                    raise ValueError(
                        "Client model-state tensor shape "
                        f"mismatch: {name}"
                    )

        aggregated_state = OrderedDict()

        for name in reference_keys:
            reference_tensor = reference_state[name]

            if not (
                torch.is_floating_point(
                    reference_tensor
                )
                or torch.is_complex(
                    reference_tensor
                )
            ):
                for result in client_results[1:]:
                    if not torch.equal(
                        result.state[name],
                        reference_tensor,
                    ):
                        raise ValueError(
                            "Non-floating model state differs "
                            f"across clients: {name}"
                        )

                aggregated_state[name] = (
                    reference_tensor
                    .detach()
                    .cpu()
                    .clone()
                )

                continue

            accumulator = torch.zeros_like(
                reference_tensor,
                device="cpu",
            )

            for result in client_results:
                accumulator.add_(
                    result.state[name]
                    .detach()
                    .to("cpu"),
                    alpha=weights[
                        result.client_id
                    ],
                )

            aggregated_state[name] = accumulator

        return aggregated_state
