from collections import OrderedDict

import torch


def get_model_state(model, device="cpu", clone=True):
    state = OrderedDict()

    for name, tensor in model.state_dict().items():
        value = tensor.detach().to(device)

        if clone:
            value = value.clone()

        state[name] = value

    return state


def load_model_state(model, state, strict=True):
    model.load_state_dict(state, strict=strict)
    return model


def clone_model_state(state, device="cpu"):
    return OrderedDict(
        (
            name,
            tensor.detach().to(device).clone(),
        )
        for name, tensor in state.items()
    )


def model_states_equal(state_a, state_b):
    if state_a.keys() != state_b.keys():
        return False

    return all(
        torch.equal(state_a[name], state_b[name])
        for name in state_a
    )
