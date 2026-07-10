from collections import Counter


def relation_counts_from_data(
    data,
):
    if not hasattr(data, "edge_type"):
        raise ValueError(
            "Training data is missing edge_type"
        )

    edge_types = (
        data.edge_type
        .detach()
        .cpu()
        .tolist()
    )

    if not edge_types:
        raise ValueError(
            "Training data contains no edges"
        )

    return dict(
        sorted(
            Counter(edge_types).items()
        )
    )


def relation_distribution_from_data(
    data,
):
    counts = relation_counts_from_data(
        data
    )

    total = sum(counts.values())

    return {
        relation_id:
            count / total
        for relation_id, count
        in counts.items()
    }


def build_client_relation_distributions(
    enterprises,
    client_ids,
):
    client_ids = tuple(client_ids)

    if not client_ids:
        raise ValueError(
            "At least one client is required"
        )

    if len(client_ids) != len(
        set(client_ids)
    ):
        raise ValueError(
            "Client IDs must be unique"
        )

    distributions = {}

    for client_id in client_ids:
        if client_id not in enterprises:
            raise ValueError(
                "Missing enterprise data: "
                f"{client_id}"
            )

        distributions[client_id] = (
            relation_distribution_from_data(
                enterprises[
                    client_id
                ]["train_data"]
            )
        )

    return distributions
