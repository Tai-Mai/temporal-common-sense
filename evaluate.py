import fire
import json
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import plotly.figure_factory as ff

import plotly.io as pio

pio.kaleido.scope.mathjax = None


def main(filepath: str):
    filetype = "pdf"
    matrix_dir = Path(filepath.split("/")[0])
    plot_name = filepath.split("/")[-1].split(".")[0]
    model_name = plot_name.split("_")[0]
    normalized = "normalized" in plot_name

    with open(filepath, "r") as f:
        confusion_matrix_dict = json.load(f)
    # ic(confusion_matrix_dict)

    confusion_matrix: list[list[float]] = [
        list(counts.values()) for counts in confusion_matrix_dict.values()
    ]
    confusion_matrix.reverse()

    if "values" in plot_name:
        confusion_array: ArrayLike = np.array(confusion_matrix)
        means = np.mean(confusion_array, axis=-1).tolist()
        confusion_matrix = means
        standard_deviations = np.std(confusion_array, axis=-1).tolist()
        annotations = []
        for mean_row, std_row in zip(means, standard_deviations):
            row_annotations: list[str] = []
            for mean, standard_deviation in zip(mean_row, std_row):
                row_annotations.append(f"{mean:.2f} ± {standard_deviation:.2f}")
            annotations.append(row_annotations)
    else:
        annotations = None

    relations: list[str] = list(confusion_matrix_dict.keys())

    fig = ff.create_annotated_heatmap(
        confusion_matrix,
        x=relations,
        y=relations[::-1],
        annotation_text=annotations,
        reversescale=True if "values" in plot_name else False,
    )
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 8

    plot_dir = matrix_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig.update_layout(
        # title={"text": f"{model_name}{' normalized' if normalized else ''}"},
        margin=dict(l=0, r=0, t=0, b=0),
        font={"size": 20},
    )
    fig.write_image(plot_dir / (plot_name + f".{filetype}"))


if __name__ == "__main__":
    fire.Fire(main)
