import fire
import json
from pathlib import Path
import plotly.express as px
import scipy.stats
import statistics

import plotly.io as pio

pio.kaleido.scope.mathjax = None

CONCEPTUAL_NEIGHBORHOOD: dict[str, dict[str, int]] = {
    "before": {
        "before": 0,
        "meets": 1,
        "overlaps": 2,
        "starts": 3,
        "during": 4,
        "finishes": 5,
        "equals": 4,
    },
    "meets": {
        "before": 1,
        "meets": 0,
        "overlaps": 1,
        "starts": 2,
        "during": 3,
        "finishes": 4,
        "equals": 3,
    },
    "overlaps": {
        "before": 2,
        "meets": 1,
        "overlaps": 0,
        "starts": 1,
        "during": 2,
        "finishes": 3,
        "equals": 4,
    },
    "starts": {
        "before": 3,
        "meets": 2,
        "overlaps": 1,
        "starts": 0,
        "during": 1,
        "finishes": 2,
        "equals": 1,
    },
    "during": {
        "before": 4,
        "meets": 3,
        "overlaps": 2,
        "starts": 1,
        "during": 0,
        "finishes": 1,
        "equals": 1,
    },
    "finishes": {
        "before": 5,
        "meets": 4,
        "overlaps": 3,
        "starts": 2,
        "during": 1,
        "finishes": 0,
        "equals": 1,
    },
    "equals": {
        "before": 4,
        "meets": 3,
        "overlaps": 2,
        "starts": 1,
        "during": 1,
        "finishes": 1,
        "equals": 0,
    },
}


def main(filepath: str, deltas: bool = False):
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_name = filepath.split("/")[-1].split(".")[0]

    with open(filepath, "r") as f:
        confusion_matrix_dict: dict[str, dict[str, list[float]]] = json.load(f)

    hops: list[int] = []
    perplexities: list[float] = []

    for true_relation, connections in confusion_matrix_dict.items():
        average_true_perplexity: float = statistics.fmean(connections[true_relation])

        for predicted_relation, predicted_perplexities in connections.items():
            if deltas:
                perplexities += [
                    ppl - average_true_perplexity for ppl in predicted_perplexities
                ]
            else:
                perplexities += predicted_perplexities
            hops += [CONCEPTUAL_NEIGHBORHOOD[true_relation][predicted_relation]] * len(
                predicted_perplexities
            )

    pearson_coefficient = scipy.stats.pearsonr(hops, perplexities)
    spearman_coefficient = scipy.stats.spearmanr(hops, perplexities)
    kendall_coefficient = scipy.stats.kendalltau(hops, perplexities)

    print(
        f"Correlation coefficients between {'perplexity deltas' if deltas else 'raw perplexities'} and graph hops"
    )
    print("Pearson\tSpearman\tKendall")
    print(
        f"{pearson_coefficient[0]:.4f}\t{spearman_coefficient[0]:.4f}\t\t{kendall_coefficient[0]:.4f}"
    )
    print("p-values")
    print(
        f"{pearson_coefficient[1]:.4f}\t{spearman_coefficient[1]:.4f}\t\t{kendall_coefficient[1]:.4f}"
    )

    fig = px.scatter(
        x=hops,
        y=perplexities,
        # trendline="ols",
        # marginal_x="histogram",
        # marginal_y="histogram",
        labels={
            "x": "Graph hops",
            "y": f"Perplexity{' deltas' if deltas else ''}{' (normalized)' if 'normalized' in plot_name else ''}",
        },
        log_y=True,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        font={"size": 20},
    )
    fig.write_image(
        plot_dir
        / (
            plot_name
            + f"{'_deltas' if deltas else ''}{'_normalized' if 'normalized' in plot_name else ''}_correlation.pdf"
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
