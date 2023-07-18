import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union


def plot_results(
    y: pd.Series,
    y_hat,
    title: str,
    r2_score: Union[float, np.float64],
    save: bool = True,
):
    """
    Plots the results of a model.

    Params:
    - `y` : array-like - The actual values.
    - `y_hat` : array-like - The predicted values.
    - `title` : str - The title of the plot.
    - `save` : bool - Whether to save the plot as a png. Default is False.
    """

    if len(y) != len(y_hat):
        raise ValueError("y and y_hat must be the same length")

    # Create a figure and a set of subplots
    _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(y, y_hat, ".")
    ax.plot(y, y, linestyle=":")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    # Add the correlation of determination (R2) to the plot
    ax.text(0.8, 0.1, f"R2: {r2_score:.2f}", transform=ax.transAxes)

    if save:
        snake_case_title = title.replace(" ", "_")
        file_path = f"models/plots/{snake_case_title}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()