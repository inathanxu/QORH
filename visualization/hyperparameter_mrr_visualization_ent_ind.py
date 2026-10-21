from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "SimHei"]
    plt.rcParams["font.size"] = 20
except ImportError as error:
    raise SystemExit(
        "Missing plotting dependencies. Please run the following command to install: "
        "python -m pip install matplotlib pandas openpyxl numpy"
    ) from error


DEFAULT_INPUT = Path(
    "260817-hyperparameters-maxRules-maxLen-traditional-inductive-datasets-copy-for-processing.xlsx"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "mrr_visualization_results"
DEFAULT_DATASET_NAMES = (
    "WN18RR-v1",
    "WN18RR-v2",
    "WN18RR-v3",
    "WN18RR-v4",
    "FB15k-237-v1",
    "FB15k-237-v2",
    "FB15k-237-v3",
    "FB15k-237-v4",
    "NELL-995-v1",
    "NELL-995-v2",
    "NELL-995-v3",
    "NELL-995-v4",
)

PARAMETER_PATTERN = re.compile(
    r"maxRules\s*=?\s*(?P<max_rules>\d+).*?maxLen\s*=?\s*(?P<max_len>\d+)",
    flags=re.IGNORECASE,
)
PARAMETER_SEARCH_PATTERN = re.compile(
    r"maxRules\s*=?\s*\d+.*?maxLen\s*=?\s*\d+", flags=re.IGNORECASE
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the effect of maxRules and maxLen on MRR."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source .xlsx file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and CSV summary results are saved.",
    )
    parser.add_argument(
        "--sheet",
        type=int,
        default=0,
        help="Index of the worksheet to read, counted from 0, default 0.",
    )
    parser.add_argument(
        "--dataset-names",
        default=None,
        help=(
            "Optional: dataset names separated by commas. If not provided, "
            "12 columns default in order to WN18RR-v1 through v4, "
            "FB15k-237-v1 through v4, NELL-995-v1 through v4; "
            "other column counts use Dataset 1, Dataset 2, ..."
        ),
    )
    return parser.parse_args()


def load_mrr_table(
    input_path: Path, sheet: int, dataset_names_arg: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    raw = pd.read_excel(input_path, sheet_name=sheet, header=None)
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")

    pattern_hits = raw.apply(
        lambda column: column.fillna("").astype(str).str.contains(
            PARAMETER_SEARCH_PATTERN, na=False
        ).sum()
    )
    combination_column = pattern_hits.idxmax()
    if pattern_hits.loc[combination_column] < 2:
        raise ValueError(
            "Could not find at least two combination names matching the "
            "maxRules...maxLen format in the workbook."
        )

    combination_text = raw[combination_column].fillna("").astype(str).str.strip()
    valid_rows = combination_text.str.contains(PARAMETER_SEARCH_PATTERN, na=False)
    combinations = combination_text.loc[valid_rows].reset_index(drop=True)

    metric_columns: list[object] = []
    for column in raw.columns:
        if column == combination_column:
            continue
        numeric = pd.to_numeric(raw.loc[valid_rows, column], errors="coerce")
        if numeric.notna().all():
            metric_columns.append(column)

    if not metric_columns:
        raise ValueError("Could not find an MRR column that is numeric for all hyperparameter combinations.")

    scores = (
        raw.loc[valid_rows, metric_columns]
        .apply(pd.to_numeric, errors="raise")
        .reset_index(drop=True)
    )

    if dataset_names_arg:
        dataset_names = [name.strip() for name in dataset_names_arg.split(",")]
        if len(dataset_names) != scores.shape[1] or any(not name for name in dataset_names):
            raise ValueError(
                f"--dataset-names must provide exactly {scores.shape[1]} non-empty dataset names."
            )
    elif scores.shape[1] == len(DEFAULT_DATASET_NAMES):
        dataset_names = list(DEFAULT_DATASET_NAMES)
    else:
        dataset_names = [f"Dataset {index}" for index in range(1, scores.shape[1] + 1)]
    scores.columns = dataset_names

    parameters = combinations.str.extract(PARAMETER_PATTERN).astype(int)
    parameters.insert(0, "combination", combinations)
    return parameters, scores


def summarize(parameters: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_by_dataset = scores.max(axis=0)
    best_mask = pd.DataFrame(
        np.isclose(scores.to_numpy(), best_by_dataset.to_numpy()[None, :], atol=1e-12),
        index=scores.index,
        columns=scores.columns,
    )

    summary = parameters.copy()
    summary["mean_mrr"] = scores.mean(axis=1)
    summary["median_mrr"] = scores.median(axis=1)
    summary["std_mrr"] = scores.std(axis=1, ddof=1)
    summary["min_mrr"] = scores.min(axis=1)
    summary["max_mrr"] = scores.max(axis=1)
    summary["best_dataset_count"] = best_mask.sum(axis=1)
    summary["mean_rank"] = scores.rank(axis=0, ascending=False, method="average").mean(axis=1)
    summary["plot_label"] = [
        f"maxRules{rules}, maxLen{length}"
        for rules, length in zip(summary["max_rules"], summary["max_len"])
    ]

    return summary, best_mask


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(
    output_path.with_suffix(".pdf"),
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
    plt.close(figure)


def get_logical_order(summary: pd.DataFrame) -> pd.Index:
    return summary.sort_values(["max_len", "max_rules"], kind="stable").index


def plot_mrr_lines(scores: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    order = get_logical_order(summary)
    x = np.arange(scores.shape[1])

    color_maps = {2: plt.cm.Blues, 3: plt.cm.Oranges}
    color_by_index: dict[object, tuple[float, float, float, float]] = {}
    for length in sorted(summary["max_len"].unique()):
        indices = summary.loc[summary["max_len"] == length].sort_values("max_rules").index
        color_map = color_maps.get(length, plt.cm.Greys)
        shades = np.linspace(0.45, 0.90, len(indices))
        color_by_index.update(dict(zip(indices, color_map(shades))))

    figure_height = max(6.8, 0.38 * len(order) + 1.6)
    figure, ax = plt.subplots(figsize=(13.5, figure_height))
    for index in order:
        label = summary.loc[index, "plot_label"]
        ax.plot(
            x,
            scores.loc[index].to_numpy(),
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color_by_index[index],
            label=label,
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("MRR")

    versions = [name.rsplit("-", 1)[-1] for name in scores.columns]
    ax.set_xticks(x, versions, rotation=0, ha="center", fontsize=16)

    for center, name in zip(
        [1.5, 5.5, 9.5],
        ["WN18RR", "FB15k-237", "NELL-995"],
    ):
        ax.text(
            center, -0.07, name,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=16,
        )
    ax.set_xlabel("Dataset", labelpad=25)

    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        title="Combination",
        bbox_to_anchor=(1.0, 1),
        loc="upper left",
        frameon=False,
        ncol=1,
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "01_mrr_line_by_dataset.png")


def plot_winner_counts(summary: pd.DataFrame, output_dir: Path) -> None:
    display = summary.loc[get_logical_order(summary)].reset_index(drop=True)
    unique_lengths = sorted(display["max_len"].unique())
    length_colors = {
        length: plt.cm.Set2(position)
        for length, position in zip(unique_lengths, np.linspace(0, 1, len(unique_lengths)))
    }
    colors = [length_colors[length] for length in display["max_len"]]

    figure_width = max(12.5, 0.85 * len(display) + 2)
    figure, ax = plt.subplots(figsize=(figure_width, 6.2))
    bars = ax.bar(display["plot_label"], display["best_dataset_count"], color=colors)
    ax.bar_label(bars, padding=3, fmt="%.0f")

    ax.set_xlabel("Hyperparameter combination")
    ax.set_ylabel("Number of highest-MRR data sets")
    ax.set_ylim(0, max(1, display["best_dataset_count"].max()) + 0.7)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor=length_colors[length], label=f"maxLen = {length}")
            for length in unique_lengths
        ],
        title="Color encodes",
        frameon=False,
    )
    ax.text(
        0.01,
        -0.23,
        "If values tie for a data set, every tied configuration receives one win.",
        transform=ax.transAxes,
        color="dimgray",
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "02_best_count_by_combination.png")


def plot_score_heatmap(
    scores: pd.DataFrame, summary: pd.DataFrame, best_mask: pd.DataFrame, output_dir: Path
) -> None:
    order = get_logical_order(summary)
    display_scores = scores.loc[order]
    display_mask = best_mask.loc[order]
    labels = summary.loc[order, "plot_label"].to_list()

    figure, ax = plt.subplots(figsize=(14, max(6.0, 0.52 * len(order) + 1.9)))
    image = ax.imshow(display_scores.to_numpy(), aspect="auto", cmap="YlGnBu")
    colorbar = figure.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("MRR")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Hyperparameter combination")
    ax.set_xticks(np.arange(scores.shape[1]), scores.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)

    for row in range(display_scores.shape[0]):
        for column in range(display_scores.shape[1]):
            value = display_scores.iat[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=10)
            if display_mask.iat[row, column]:
                ax.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#f4b400",
                        linewidth=2.4,
                    )
                )

    figure.tight_layout()
    save_figure(figure, output_dir / "03_mrr_heatmap.png")


def plot_sensitivity(summary: pd.DataFrame, output_dir: Path) -> None:
    rule_values = sorted(summary["max_rules"].unique())
    length_values = sorted(summary["max_len"].unique())
    colors = plt.cm.Dark2(np.linspace(0, 1, len(length_values)))

    figure, (line_ax, heatmap_ax) = plt.subplots(1, 2, figsize=(14, 5.8))
    for color, length in zip(colors, length_values):
        subset = summary.loc[summary["max_len"] == length].sort_values("max_rules")
        line_ax.errorbar(
            subset["max_rules"],
            subset["mean_mrr"],
            yerr=subset["std_mrr"],
            marker="o",
            linewidth=2,
            capsize=4,
            color=color,
            label=f"maxLen = {length}",
        )
    line_ax.set_title("Mean MRR vs. maxRules (error bar = 1 SD)",fontsize=18)
    line_ax.set_xlabel("maxRules")
    line_ax.set_ylabel("Mean MRR across datasets")
    line_ax.set_xticks(rule_values)
    line_ax.grid(alpha=0.3)
    line_ax.legend(
    loc="lower left",
    bbox_to_anchor=(0.12, 0.02),
    frameon=False,
    borderaxespad=0,)

    grid = (
        summary.pivot(index="max_len", columns="max_rules", values="mean_mrr")
        .reindex(index=length_values, columns=rule_values)
    )
    image = heatmap_ax.imshow(grid.to_numpy(), cmap="viridis", aspect="auto")
    figure.colorbar(image, ax=heatmap_ax, pad=0.02, label="Mean MRR")
    heatmap_ax.set_title("Mean MRR Parameter Grid",fontsize=18)
    heatmap_ax.set_xlabel("maxRules")
    heatmap_ax.set_ylabel("maxLen")
    heatmap_ax.set_xticks(np.arange(len(rule_values)), rule_values)
    heatmap_ax.set_yticks(np.arange(len(length_values)), length_values)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            heatmap_ax.text(
                column,
                row,
                f"{grid.iat[row, column]:.3f}",
                ha="center",
                va="center",
                color="white",
            )

    figure.tight_layout()
    save_figure(figure, output_dir / "04_hyperparameter_sensitivity.png")


def export_tables(
    summary: pd.DataFrame, scores: pd.DataFrame, best_mask: pd.DataFrame, output_dir: Path
) -> None:
    summary_columns = [
        "combination",
        "max_rules",
        "max_len",
        "mean_mrr",
        "median_mrr",
        "std_mrr",
        "min_mrr",
        "max_mrr",
        "best_dataset_count",
        "mean_rank",
    ]
    summary.loc[:, summary_columns].sort_values(
        ["mean_mrr", "mean_rank"], ascending=[False, True]
    ).to_csv(output_dir / "hyperparameter_summary.csv", index=False, encoding="utf-8-sig")

    best_values = scores.max(axis=0)
    winner_rows = []
    for dataset in scores.columns:
        winners = summary.loc[best_mask[dataset], "combination"].to_list()
        winner_rows.append(
            {
                "dataset": dataset,
                "best_mrr": best_values[dataset],
                "winning_combination(s)": "; ".join(winners),
            }
        )
    pd.DataFrame(winner_rows).to_csv(
        output_dir / "best_combination_by_dataset.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters, scores = load_mrr_table(args.input, args.sheet, args.dataset_names)
    summary, best_mask = summarize(parameters, scores)

    plot_mrr_lines(scores, summary, args.output_dir)
    plot_winner_counts(summary, args.output_dir)
    plot_score_heatmap(scores, summary, best_mask, args.output_dir)
    plot_sensitivity(summary, args.output_dir)
    export_tables(summary, scores, best_mask, args.output_dir)

    best = summary.sort_values(["mean_mrr", "mean_rank"], ascending=[False, True]).iloc[0]
    print(f"Best overall combination by mean MRR: {best['combination']}")
    print(f"  Mean MRR = {best['mean_mrr']:.4f}")
    print(f"  Mean rank = {best['mean_rank']:.2f}")
    print(f"  Times achieving a dataset's highest value = {best['best_dataset_count']}")
    print(f"Figures and CSV summary tables saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
