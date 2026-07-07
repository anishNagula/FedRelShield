from dataclasses import dataclass

from ultra import model_state


@dataclass(frozen=True)
class FederatedRoundResult:
    round_id: int
    client_results: tuple
    global_state: dict
    evaluation_result: object = None


class FederatedCoordinator:
    def __init__(
        self,
        server,
        clients,
        aggregator,
        evaluator=None,
    ):
        self.server = server
        self.clients = tuple(clients)
        self.aggregator = aggregator
        self.evaluator = evaluator

        if len(self.clients) < 2:
            raise ValueError(
                "FederatedCoordinator requires "
                "at least two clients"
            )

        client_ids = [
            client.client_id
            for client in self.clients
        ]

        if len(client_ids) != len(set(client_ids)):
            raise ValueError(
                "Federated client IDs must be unique"
            )

    def run_round(
        self,
        round_id,
        local_epochs,
        batch_per_epoch,
    ):
        if round_id < 0:
            raise ValueError(
                "round_id must be non-negative"
            )

        broadcast_state = (
            self.server.broadcast_state()
        )

        client_results = tuple(
            client.train(
                global_state=broadcast_state,
                local_epochs=local_epochs,
                batch_per_epoch=batch_per_epoch,
                round_id=round_id,
            )
            for client in self.clients
        )

        for result in client_results:
            if model_state.model_states_equal(
                broadcast_state,
                result.state,
            ):
                raise RuntimeError(
                    "Client local state did not change: "
                    f"{result.client_id}"
                )

        aggregated_state = self.aggregator.aggregate(
            client_results
        )

        for result in client_results:
            if model_state.model_states_equal(
                aggregated_state,
                result.state,
            ):
                raise RuntimeError(
                    "Aggregated state unexpectedly equals "
                    f"client state: {result.client_id}"
                )

        self._validate_aggregate_contract(
            broadcast_state=broadcast_state,
            aggregated_state=aggregated_state,
        )

        self.server.load_global_state(
            aggregated_state
        )

        global_state = (
            self.server.get_global_state()
        )

        if not model_state.model_states_equal(
            aggregated_state,
            global_state,
        ):
            raise RuntimeError(
                "Server state does not match "
                "aggregated state after loading"
            )

        evaluation_result = None

        if self.evaluator is not None:
            evaluation_result = (
                self.server.evaluate(
                    self.evaluator
                )
            )

        return FederatedRoundResult(
            round_id=round_id,
            client_results=client_results,
            global_state=global_state,
            evaluation_result=evaluation_result,
        )

    def run(
        self,
        num_rounds,
        local_epochs,
        batch_per_epoch,
        start_round=0,
    ):
        if num_rounds <= 0:
            raise ValueError(
                "num_rounds must be positive"
            )

        if start_round < 0:
            raise ValueError(
                "start_round must be non-negative"
            )

        results = []

        for round_id in range(
            start_round,
            start_round + num_rounds,
        ):
            results.append(
                self.run_round(
                    round_id=round_id,
                    local_epochs=local_epochs,
                    batch_per_epoch=batch_per_epoch,
                )
            )

        return tuple(results)

    def _validate_aggregate_contract(
        self,
        broadcast_state,
        aggregated_state,
    ):
        if (
            broadcast_state.keys()
            != aggregated_state.keys()
        ):
            raise RuntimeError(
                "Aggregated state keys do not match "
                "global model state"
            )

        for name in broadcast_state:
            if (
                broadcast_state[name].shape
                != aggregated_state[name].shape
            ):
                raise RuntimeError(
                    "Aggregated state shape mismatch: "
                    f"{name}"
                )
