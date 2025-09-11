"""Radar plots for GP libraries comparison."""

from math import pi
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .utils import save
from ..metrics import GPlibrary


def _adimensional_metrics_by_library(gp_libs: dict[str, "GPlibrary"], path: Path) -> None:
    libraries, _ = gp_libs.keys(), len(gp_libs.keys())

    MSE = np.array([gp_libs[lib].mse for lib in gp_libs])
    RMSE = np.array([gp_libs[lib].rmse for lib in gp_libs])
    MAE = np.array([gp_libs[lib].mae for lib in gp_libs])
    R2 = np.array([gp_libs[lib].r2 for lib in gp_libs])
    MedAE = np.array([gp_libs[lib].medae for lib in gp_libs])
    NLPD = np.array([gp_libs[lib].nlpd for lib in gp_libs])
    MSLL = np.array([gp_libs[lib].msll for lib in gp_libs])
    train_time = np.array([gp_libs[lib].train_time for lib in gp_libs])
    pred_time = np.array([gp_libs[lib].pred_time for lib in gp_libs])
    train_memory = np.array([gp_libs[lib].train_memory for lib in gp_libs])
    pred_memory = np.array([gp_libs[lib].pred_memory for lib in gp_libs])

    data = [
        MSE / MSE.max(),
        RMSE / RMSE.max(),
        MAE / MAE.max(),
        R2,
        MedAE / MedAE.max(),
        np.abs(NLPD) / np.abs(NLPD).max(),
        np.abs(MSLL) / np.abs(MSLL).max(),
        train_time / train_time.max(),
        pred_time / pred_time.max(),
        train_memory / train_memory.max(),
        pred_memory / pred_memory.max(),
    ]

    data_labels = [
        r"$MSE^*$",
        r"$RMSE^*$",
        r"$MAE^*$",
        r"$R^{2^*}$",
        r"$MedAE^*$",
        r"$NLPD^*$",
        r"$MSLL^*$",
        r"$t_{\text{train}}^*$",
        r"$t_{\text{pred}}^*$",
        r"$\text{mem.}_{\text{train}}^*$",
        r"$\text{mem.}_{\text{pred}}^*$",
    ]
    num_metrics = len(data_labels)

    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    for lib_name, lib_data in zip(
        libraries,
        np.array(data).T,
        strict=False,
    ):
        lib_data = list(lib_data)
        lib_data += lib_data[:1]
        ax.plot(angles, lib_data, linewidth=0.75, linestyle="solid", label=lib_name)
        ax.fill(angles, lib_data, alpha=0.1)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(data_labels)
    ax.tick_params(axis="x", pad=17.5)

    plt.legend(loc="upper right", bbox_to_anchor=(1.55, 1.0))
    ax.grid(True)
    plt.tight_layout()  # Adjust the rect parameter to make room for legend
    save(plt, path=path, filename="radar_by_library")
    plt.close()


def _metrics(gp_libs: dict, path: Path) -> None:
    metrics = [
        "MAE",
        "RMSE",
        "MSE",
        "MedAE",
        "R2",
        "NLPD",
        "MSLL",
        "train_time",
        "pred_time",
        "train_memory",
        "pred_memory",
    ]
    libraries = gp_libs.keys()
    num_libs = len(libraries)

    for metric in metrics:
        data = [[getattr(gp_libs[lib], metric.lower()) for lib in gp_libs]]

        match metric:
            case "R2":
                data_labels = [r"$R^2$"]
            case "train_time":
                data_labels = [r"$t_{\text{train}}$"]
            case "pred_time":
                data_labels = [r"$t_{\text{pred}}$"]
            case "train_memory":
                data_labels = [r"$\text{Memory}_{\text{train}}\,\,\text{(MB)}$"]
            case "pred_memory":
                data_labels = [r"$\text{Memory}_{\text{pred}}\,\,\text{(MB)}$"]
            case _:
                data_labels = [rf"${metric}$"]

        # Compute angle for each category
        angles = [n / float(num_libs) * 2 * pi for n in range(num_libs)]
        angles += angles[:1]  # Close the circle

        # Create the radar chart
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        # Plot each country's data
        for idx, (label, lib_data) in enumerate(zip(data_labels, data, strict=False)):
            lib_data += lib_data[:1]  # Close the circle for each country
            ax.plot(angles, lib_data, linewidth=0.75, linestyle="solid", label=label)
            ax.fill(angles, lib_data, alpha=0.2)

        # Customize the plot
        ax.set_theta_offset(pi / 2)  # Rotate the chart
        ax.set_theta_direction(-1)  # Inverse the plot direction
        ax.set_rlabel_position(0)  # Move radial labels away from plotted line
        ax.set_xticks(angles[:-1])  # Set category labels
        ax.set_xticklabels(libraries)
        ax.tick_params(axis="x", pad=17.5)

        # Set the range of values for radial axis
        ax.set_rlabel_position(-0.0)  # Move labels outside
        plt.yticks(color="grey", size=10)

        # Add the legend
        plt.legend(loc="upper right", bbox_to_anchor=(1.43, 1.1))
        ax.grid(True)
        plt.tight_layout()  # Adjust the rect parameter to make room for legend

        save(plt, path=path, filename=f"radar_{metric}")
        plt.close()
