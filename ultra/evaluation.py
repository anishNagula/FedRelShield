import math

import torch
from torch import distributed as dist

from ultra import tasks, util


@torch.no_grad()
def evaluate(
    cfg,
    model,
    eval_data,
    device,
    filtered_data=None,
    logger=None,
    return_metrics=False,
):
    world_size = util.get_world_size()
    rank = util.get_rank()

    eval_triplets = torch.cat(
        [
            eval_data.target_edge_index,
            eval_data.target_edge_type.unsqueeze(0),
        ]
    ).t()

    sampler = torch.utils.data.DistributedSampler(
        eval_triplets,
        world_size,
        rank,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_triplets,
        cfg.train.batch_size,
        sampler=sampler,
    )

    model.eval()

    rankings = []
    num_negatives = []

    tail_rankings = []
    num_tail_negs = []

    for batch in eval_loader:
        t_batch, h_batch = tasks.all_negative(
            eval_data,
            batch,
        )

        t_pred = model(eval_data, t_batch)
        h_pred = model(eval_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(
                eval_data,
                batch,
            )
        else:
            t_mask, h_mask = tasks.strict_negative_mask(
                filtered_data,
                batch,
            )

        pos_h_index, pos_t_index, pos_r_index = batch.t()

        t_ranking = tasks.compute_ranking(
            t_pred,
            pos_t_index,
            t_mask,
        )

        h_ranking = tasks.compute_ranking(
            h_pred,
            pos_h_index,
            h_mask,
        )

        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings.append(t_ranking)
        num_tail_negs.append(num_t_negative)

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)

    tail_ranking = torch.cat(tail_rankings)
    num_tail_negative = torch.cat(num_tail_negs)

    all_size = torch.zeros(
        world_size,
        dtype=torch.long,
        device=device,
    )
    all_size[rank] = len(ranking)

    all_tail_size = torch.zeros(
        world_size,
        dtype=torch.long,
        device=device,
    )
    all_tail_size[rank] = len(tail_ranking)

    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_tail_size, op=dist.ReduceOp.SUM)

    cumulative_size = all_size.cumsum(0)

    all_ranking = torch.zeros(
        all_size.sum(),
        dtype=torch.long,
        device=device,
    )

    all_num_negative = torch.zeros(
        all_size.sum(),
        dtype=torch.long,
        device=device,
    )

    start = cumulative_size[rank] - all_size[rank]
    end = cumulative_size[rank]

    all_ranking[start:end] = ranking
    all_num_negative[start:end] = num_negative

    cumulative_tail_size = all_tail_size.cumsum(0)

    all_tail_ranking = torch.zeros(
        all_tail_size.sum(),
        dtype=torch.long,
        device=device,
    )

    all_tail_num_negative = torch.zeros(
        all_tail_size.sum(),
        dtype=torch.long,
        device=device,
    )

    tail_start = (
        cumulative_tail_size[rank]
        - all_tail_size[rank]
    )
    tail_end = cumulative_tail_size[rank]

    all_tail_ranking[tail_start:tail_end] = tail_ranking
    all_tail_num_negative[tail_start:tail_end] = (
        num_tail_negative
    )

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_tail_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(
            all_tail_num_negative,
            op=dist.ReduceOp.SUM,
        )

    metrics = {}

    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                metric_name, direction = metric.split("-")

                if direction != "tail":
                    raise ValueError(
                        "Only tail metric is supported in this mode"
                    )

                metric_ranking = all_tail_ranking
                metric_num_negative = all_tail_num_negative

            else:
                metric_name = metric
                metric_ranking = all_ranking
                metric_num_negative = all_num_negative

            if metric_name == "mr":
                score = metric_ranking.float().mean()

            elif metric_name == "mrr":
                score = (
                    1 / metric_ranking.float()
                ).mean()

            elif metric_name.startswith("hits@"):
                values = metric_name[5:].split("_")
                threshold = int(values[0])

                if len(values) > 1:
                    num_sample = int(values[1])

                    fp_rate = (
                        (metric_ranking - 1).float()
                        / metric_num_negative
                    )

                    score = 0

                    for i in range(threshold):
                        num_comb = (
                            math.factorial(num_sample - 1)
                            / math.factorial(i)
                            / math.factorial(
                                num_sample - i - 1
                            )
                        )

                        score += (
                            num_comb
                            * (fp_rate ** i)
                            * (
                                (1 - fp_rate)
                                ** (num_sample - i - 1)
                            )
                        )

                    score = score.mean()

                else:
                    score = (
                        metric_ranking <= threshold
                    ).float().mean()

            else:
                raise ValueError(
                    "Unknown metric `%s`" % metric_name
                )

            if logger is not None:
                logger.warning(
                    "%s: %g" % (metric, score)
                )

            metrics[metric] = score

    mrr = (
        1 / all_ranking.float()
    ).mean()

    if return_metrics:
        return metrics

    return mrr
