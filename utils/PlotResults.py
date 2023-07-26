from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


def plot_results(
    y: pd.Series,
    y_hat,
    title: str,
    save: bool = False,
):
    """
    Plots the results of a model.

    Params:
    - `y` : array-like - The actual values.
    - `y_hat` : array-like - The predicted values.
    - `title` : str - The title of the plot.
    - `r2_score` : float - The R2 score of the model.
    - `save` : bool - Whether to save the plot as a png. Default is False.
    """

    if len(y) != len(y_hat):
        raise ValueError("y and y_hat must be the same length")

    _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(y, y_hat, ".")
    ax.plot(y, y, linestyle=":")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    score = r2_score(y, y_hat)
    ax.text(0.8, 0.1, f"R2: {score:.2f}", transform=ax.transAxes)

    if save:
        snake_case_title = title.replace(" ", "_")
        file_path = f"models/plots/{snake_case_title}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")

    plt.show()