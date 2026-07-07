from ultra import model_state


class FederatedServer:
    def __init__(
        self,
        model,
    ):
        self.model = model

    def get_global_state(self):
        return model_state.get_model_state(
            self.model,
            device="cpu",
            clone=True,
        )

    def load_global_state(
        self,
        state,
    ):
        model_state.load_model_state(
            self.model,
            state,
        )

    def broadcast_state(self):
        return model_state.get_model_state(
            self.model,
            device="cpu",
            clone=True,
        )
