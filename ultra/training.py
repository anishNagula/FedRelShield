import torch

from torch import optim
from torch.nn import functional as F
from torch.utils import data as torch_data

from ultra import tasks


def build_optimizer(cfg, model):
    optimizer_cfg = dict(cfg.optimizer)
    cls = optimizer_cfg.pop("class")

    return getattr(optim, cls)(
        model.parameters(),
        **optimizer_cfg,
    )


def build_train_loader(
    train_data,
    batch_size,
    world_size=1,
    rank=0,
):
    train_triplets = torch.cat(
        [
            train_data.target_edge_index,
            train_data.target_edge_type.unsqueeze(0),
        ]
    ).t()

    sampler = torch_data.DistributedSampler(
        train_triplets,
        num_replicas=world_size,
        rank=rank,
    )

    train_loader = torch_data.DataLoader(
        train_triplets,
        batch_size,
        sampler=sampler,
    )

    return train_loader, sampler


def compute_loss(cfg, model, train_data, batch):
    batch = tasks.negative_sampling(
        train_data,
        batch,
        cfg.task.num_negative,
        strict=cfg.task.strict_negative,
    )

    pred = model(train_data, batch)

    target = torch.zeros_like(pred)
    target[:, 0] = 1

    loss = F.binary_cross_entropy_with_logits(
        pred,
        target,
        reduction="none",
    )

    neg_weight = torch.ones_like(pred)

    if cfg.task.adversarial_temperature > 0:
        with torch.no_grad():
            neg_weight[:, 1:] = F.softmax(
                pred[:, 1:] / cfg.task.adversarial_temperature,
                dim=-1,
            )
    else:
        neg_weight[:, 1:] = 1 / cfg.task.num_negative

    loss = (
        (loss * neg_weight).sum(dim=-1)
        / neg_weight.sum(dim=-1)
    )

    return loss.mean()


def train_one_epoch(
    cfg,
    model,
    train_data,
    train_loader,
    optimizer,
    batch_per_epoch,
    on_batch_end=None,
):
    model.train()

    losses = []

    for batch_id, batch in enumerate(train_loader):
        if batch_id >= batch_per_epoch:
            break

        loss = compute_loss(
            cfg,
            model,
            train_data,
            batch,
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if on_batch_end is not None:
            on_batch_end(batch_id, loss.item())

    return sum(losses) / len(losses)
