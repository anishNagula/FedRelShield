import os
import sys
import math
import pprint

import torch
import torch_geometric as pyg
from torch import nn
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ultra import util, training, evaluation
from ultra.models import Ultra


separator = ">" * 30
line = "-" * 30


def train_and_validate(
    cfg,
    model,
    train_data,
    valid_data,
    device,
    logger,
    filtered_data=None,
    batch_per_epoch=None,
):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_loader, sampler = training.build_train_loader(
        train_data=train_data,
        batch_size=cfg.train.batch_size,
        world_size=world_size,
        rank=rank,
    )

    batch_per_epoch = batch_per_epoch or len(train_loader)

    optimizer = training.build_optimizer(cfg, model)

    num_params = sum(p.numel() for p in model.parameters())

    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
        )
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)

    best_result = float("-inf")
    best_epoch = -1

    # Preserve upstream's globally increasing batch ID used for logging.
    global_batch_id = 0

    def log_batch_loss(local_batch_id, loss):
        nonlocal global_batch_id

        if (
            util.get_rank() == 0
            and global_batch_id % cfg.train.log_interval == 0
        ):
            logger.warning(separator)
            logger.warning("binary cross entropy: %g" % loss)

        global_batch_id += 1

    for i in range(0, cfg.train.num_epoch, step):
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            sampler.set_epoch(epoch)

            avg_loss = training.train_one_epoch(
                cfg=cfg,
                model=parallel_model,
                train_data=train_data,
                train_loader=train_loader,
                optimizer=optimizer,
                batch_per_epoch=batch_per_epoch,
                on_batch_end=log_batch_loss,
            )

            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning(
                    "average binary cross entropy: %g" % avg_loss
                )

        epoch = min(cfg.train.num_epoch, i + step)

        if rank == 0:
            logger.warning(
                "Save checkpoint to model_epoch_%d.pth" % epoch
            )

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(
                state,
                "model_epoch_%d.pth" % epoch,
            )

        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")

        result = evaluation.evaluate(
            cfg=cfg,
            model=model,
            eval_data=valid_data,
            filtered_data=filtered_data,
            device=device,
            logger=logger,
        )

        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning(
            "Load checkpoint from model_epoch_%d.pth" % best_epoch
        )

    state = torch.load(
        "model_epoch_%d.pth" % best_epoch,
        map_location=device,
    )

    model.load_state_dict(state["model"])

    util.synchronize()


if __name__ == "__main__":
    args, vars = util.parse_args()

    cfg = util.load_config(
        args.config,
        context=vars,
    )

    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(
        args.seed + util.get_rank()
    )

    logger = util.get_root_logger()

    if util.get_rank() == 0:
        logger.warning(
            "Random seed: %d" % args.seed
        )

        logger.warning(
            "Config file: %s" % args.config
        )

        logger.warning(
            pprint.pformat(cfg)
        )

    task_name = cfg.task["name"]

    dataset = util.build_dataset(cfg)

    device = util.get_device(cfg)

    train_data = dataset[0].to(device)
    valid_data = dataset[1].to(device)
    test_data = dataset[2].to(device)

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if (
        "checkpoint" in cfg
        and cfg.checkpoint is not None
    ):
        state = torch.load(
            cfg.checkpoint,
            map_location="cpu",
        )

        model.load_state_dict(
            state["model"]
        )

    # model = pyg.compile(model, dynamic=True)

    model = model.to(device)

    if task_name == "InductiveInference":
        # GraIL, MTDEA and HM validation sets use the training graph.
        # ILPC and Ingram validation sets use the inference graph.

        if (
            "ILPC" in cfg.dataset["class"]
            or "Ingram" in cfg.dataset["class"]
        ):
            full_inference_edges = torch.cat(
                [
                    valid_data.edge_index,
                    valid_data.target_edge_index,
                    test_data.target_edge_index,
                ],
                dim=1,
            )

            full_inference_etypes = torch.cat(
                [
                    valid_data.edge_type,
                    valid_data.target_edge_type,
                    test_data.target_edge_type,
                ]
            )

            test_filtered_data = Data(
                edge_index=full_inference_edges,
                edge_type=full_inference_etypes,
                num_nodes=test_data.num_nodes,
            )

            val_filtered_data = test_filtered_data

        else:
            full_inference_edges = torch.cat(
                [
                    test_data.edge_index,
                    test_data.target_edge_index,
                ],
                dim=1,
            )

            full_inference_etypes = torch.cat(
                [
                    test_data.edge_type,
                    test_data.target_edge_type,
                ]
            )

            test_filtered_data = Data(
                edge_index=full_inference_edges,
                edge_type=full_inference_etypes,
                num_nodes=test_data.num_nodes,
            )

            val_filtered_data = Data(
                edge_index=torch.cat(
                    [
                        train_data.edge_index,
                        valid_data.target_edge_index,
                    ],
                    dim=1,
                ),
                edge_type=torch.cat(
                    [
                        train_data.edge_type,
                        valid_data.target_edge_type,
                    ]
                ),
            )

    else:
        filtered_data = Data(
            edge_index=dataset._data.target_edge_index,
            edge_type=dataset._data.target_edge_type,
            num_nodes=dataset[0].num_nodes,
        )

        val_filtered_data = filtered_data
        test_filtered_data = filtered_data

    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)

    train_and_validate(
        cfg=cfg,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        filtered_data=val_filtered_data,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
        logger=logger,
    )

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")

    evaluation.evaluate(
        cfg=cfg,
        model=model,
        eval_data=valid_data,
        filtered_data=val_filtered_data,
        device=device,
        logger=logger,
    )

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")

    evaluation.evaluate(
        cfg=cfg,
        model=model,
        eval_data=test_data,
        filtered_data=test_filtered_data,
        device=device,
        logger=logger,
    )
