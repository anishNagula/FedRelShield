from collections import OrderedDict

import torch


class FedAvgAggregator:
    def aggregate(
        self,
        client_results,
    ):
        client_results = tuple(client_results)

        if not client_results:
            raise ValueError(
                "FedAvg requires at least one client result"
            )

        total_examples = sum(
            result.num_examples
            for result in client_results
        )

        if total_examples <= 0:
            raise ValueError(
                "FedAvg requires positive total example count"
            )

        reference_state = client_results[0].state
        reference_keys = tuple(reference_state.keys())

        for result in client_results:
            if tuple(result.state.keys()) != reference_keys:
                raise ValueError(
                    "Client model-state keys do not match"
                )

            if result.num_examples <= 0:
                raise ValueError(
                    "Client example count must be positive"
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
                torch.is_floating_point(reference_tensor)
                or torch.is_complex(reference_tensor)
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
                    reference_tensor.detach().cpu().clone()
                )

                continue

            accumulator = torch.zeros_like(
                reference_tensor,
                device="cpu",
            )

            for result in client_results:
                weight = (
                    result.num_examples
                    / total_examples
                )

                accumulator.add_(
                    result.state[name]
                    .detach()
                    .to("cpu"),
                    alpha=weight,
                )

            aggregated_state[name] = accumulator

        return aggregated_state
