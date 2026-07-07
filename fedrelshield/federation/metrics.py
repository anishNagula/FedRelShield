from dataclasses import dataclass


@dataclass(frozen=True)
class EnterpriseEvaluationResult:
    enterprise_id: str
    num_examples: int
    metrics: dict


@dataclass(frozen=True)
class FederatedEvaluationResult:
    enterprise_results: tuple
    macro_metrics: dict
    weighted_metrics: dict


def aggregate_enterprise_metrics(
    enterprise_results,
):
    enterprise_results = tuple(enterprise_results)

    if not enterprise_results:
        raise ValueError(
            "At least one enterprise result is required"
        )

    metric_names = tuple(
        enterprise_results[0].metrics.keys()
    )

    total_examples = sum(
        result.num_examples
        for result in enterprise_results
    )

    if total_examples <= 0:
        raise ValueError(
            "Evaluation requires positive total example count"
        )

    for result in enterprise_results:
        if result.num_examples <= 0:
            raise ValueError(
                "Enterprise evaluation example count "
                "must be positive"
            )

        if tuple(result.metrics.keys()) != metric_names:
            raise ValueError(
                "Enterprise metric keys do not match"
            )

    macro_metrics = {}

    weighted_metrics = {}

    for metric_name in metric_names:
        values = [
            float(result.metrics[metric_name])
            for result in enterprise_results
        ]

        macro_metrics[metric_name] = (
            sum(values) / len(values)
        )

        weighted_metrics[metric_name] = sum(
            float(result.metrics[metric_name])
            * result.num_examples
            / total_examples
            for result in enterprise_results
        )

    return FederatedEvaluationResult(
        enterprise_results=enterprise_results,
        macro_metrics=macro_metrics,
        weighted_metrics=weighted_metrics,
    )


def evaluation_result_to_dict(result):
    return {
        "enterprises": {
            enterprise.enterprise_id: {
                "num_examples":
                    enterprise.num_examples,
                "metrics": {
                    name: float(value)
                    for name, value
                    in enterprise.metrics.items()
                },
            }
            for enterprise in result.enterprise_results
        },
        "macro_metrics": {
            name: float(value)
            for name, value
            in result.macro_metrics.items()
        },
        "weighted_metrics": {
            name: float(value)
            for name, value
            in result.weighted_metrics.items()
        },
    }
