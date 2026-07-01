from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.features import DEFAULT_FEATURE_COLUMNS as DEFAULT_FEATURES
from wm811k.paths import FEATURE_DATA_PATH, PHASE2_ANALYSIS_DIR

LABEL_COLUMN = "label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=FEATURE_DATA_PATH,
        help="Path to wafer_features.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PHASE2_ANALYSIS_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Absolute Pearson correlation threshold for redundancy removal",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=6,
        help="Number of pairwise scatter plots to export",
    )
    parser.add_argument(
        "--plot-per-class",
        type=int,
        default=500,
        help="Maximum samples per class used in scatter plots",
    )
    parser.add_argument(
        "--eval-per-class",
        type=int,
        default=None,
        help=(
            "Maximum samples per class used for quantitative evaluation. "
            "Omit this option to evaluate on every row in the feature CSV."
        ),
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, features: list[str]) -> None:
    missing = [c for c in features + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        raise KeyError(
            f"Missing columns: {missing}\n"
            f"Available numeric columns: {numeric_cols}\n"
            f"Edit DEFAULT_FEATURES if your CSV uses different feature names."
        )


def clean_data(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    work = df[features + [LABEL_COLUMN]].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[LABEL_COLUMN])

    for feature in features:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        work[feature] = work[feature].fillna(work[feature].median())

    constant = [f for f in features if work[f].nunique(dropna=False) <= 1]
    if constant:
        raise ValueError(
            f"Constant features found: {constant}. "
            "Remove them before correlation analysis."
        )

    return work


def stratified_sample(
    df: pd.DataFrame,
    max_per_class: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    if max_per_class is None:
        return df.copy()

    parts = []
    for _, group in df.groupby(LABEL_COLUMN, sort=False):
        n = min(len(group), max_per_class)
        parts.append(group.sample(n=n, random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def save_correlation_heatmap(
    corr: pd.DataFrame,
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=55, ha="right")
    ax.set_yticklabels(corr.index)

    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            value = corr.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Pearson correlation coefficient")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def get_high_correlation_pairs(
    corr: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    columns = corr.columns.tolist()

    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            value = corr.loc[left, right]
            if abs(value) >= threshold:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("absolute_correlation", ascending=False)
        .reset_index(drop=True)
    )


def calculate_mutual_information(
    df: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[LABEL_COLUMN].astype(str))
    X = df[features]

    values = mutual_info_classif(
        X,
        y,
        discrete_features=False,
        random_state=42,
    )
    return pd.Series(values, index=features, name="mutual_information").sort_values(
        ascending=False
    )


def select_nonredundant_features(
    features: list[str],
    corr: pd.DataFrame,
    high_pairs: pd.DataFrame,
    mutual_information: pd.Series,
) -> tuple[list[str], pd.DataFrame]:
    """
    Greedy rule:
    - For every pair above the correlation threshold, retain the feature with
      higher mutual information with the class label.
    - If MI is effectively equal, retain the feature with lower average
      absolute correlation to the rest of the feature set.
    """
    active = set(features)
    decisions = []
    mean_abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool)).mean()

    for _, row in high_pairs.iterrows():
        left = row["feature_1"]
        right = row["feature_2"]

        if left not in active or right not in active:
            continue

        mi_left = float(mutual_information[left])
        mi_right = float(mutual_information[right])

        if not math.isclose(mi_left, mi_right, rel_tol=1e-6, abs_tol=1e-9):
            keep = left if mi_left > mi_right else right
            drop = right if keep == left else left
            reason = "lower mutual information with label"
        else:
            keep = left if mean_abs_corr[left] <= mean_abs_corr[right] else right
            drop = right if keep == left else left
            reason = "higher mean absolute correlation with other features"

        active.remove(drop)
        decisions.append(
            {
                "kept_feature": keep,
                "dropped_feature": drop,
                "pair_correlation": row["correlation"],
                "kept_mutual_information": mutual_information[keep],
                "dropped_mutual_information": mutual_information[drop],
                "reason": reason,
            }
        )

    selected = [feature for feature in features if feature in active]
    return selected, pd.DataFrame(decisions)


def save_single_scatter(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, group in df.groupby(LABEL_COLUMN, sort=True):
        ax.scatter(
            group[x_feature],
            group[y_feature],
            s=18,
            alpha=0.55,
            label=str(label),
        )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"{x_feature} vs {y_feature}")
    ax.legend(title="Failure type", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_projection_scatter(
    coords: np.ndarray,
    labels: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            LABEL_COLUMN: labels.to_numpy(),
        }
    )

    for label, group in plot_df.groupby(LABEL_COLUMN, sort=True):
        ax.scatter(
            group["x"],
            group["y"],
            s=18,
            alpha=0.55,
            label=str(label),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Failure type", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def calculate_fisher_scores(
    df: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    """
    Multiclass Fisher score:
    between-class variance / within-class variance.
    Larger values indicate stronger univariate class separation.
    """
    scores = {}
    overall = df[features].mean()

    for feature in features:
        numerator = 0.0
        denominator = 0.0

        for _, group in df.groupby(LABEL_COLUMN):
            n = len(group)
            group_mean = group[feature].mean()
            group_var = group[feature].var(ddof=0)

            numerator += n * (group_mean - overall[feature]) ** 2
            denominator += n * group_var

        scores[feature] = numerator / (denominator + 1e-12)

    return pd.Series(scores, name="fisher_score").sort_values(ascending=False)


def evaluate_separability(
    df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    X = df[features].to_numpy()
    y_text = df[LABEL_COLUMN].astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    full_silhouette = silhouette_score(
        X_scaled,
        y,
        metric="euclidean",
        sample_size=min(5000, len(X_scaled)),
        random_state=42,
    )

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_silhouette = silhouette_score(
        X_pca,
        y,
        metric="euclidean",
        sample_size=min(5000, len(X_pca)),
        random_state=42,
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    macro_f1_scores = cross_val_score(
        classifier,
        X_scaled,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    return {
        "silhouette_full_space": float(full_silhouette),
        "silhouette_pca_2d": float(pca_silhouette),
        "random_forest_macro_f1_mean": float(macro_f1_scores.mean()),
        "random_forest_macro_f1_std": float(macro_f1_scores.std()),
        "pca_explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scatter_dir = args.output / "scatter_plots"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    validate_columns(df, DEFAULT_FEATURES)
    clean_df = clean_data(df, DEFAULT_FEATURES)

    print(f"Rows used: {len(clean_df):,}")
    print(f"Classes: {clean_df[LABEL_COLUMN].nunique()}")
    print(clean_df[LABEL_COLUMN].value_counts())

    # 1. Correlation matrix for all 12 features
    corr_all = clean_df[DEFAULT_FEATURES].corr(method="pearson")
    corr_all.to_csv(args.output / "correlation_matrix_all.csv")
    save_correlation_heatmap(
        corr_all,
        "Pearson correlation matrix - 12 handcrafted features",
        args.output / "01_correlation_matrix_all.png",
    )

    # 2. High-correlation pairs
    high_pairs = get_high_correlation_pairs(corr_all, args.threshold)
    high_pairs.to_csv(args.output / "02_high_correlation_pairs.csv", index=False)

    # 3. Preserve the more label-informative feature
    mutual_information = calculate_mutual_information(clean_df, DEFAULT_FEATURES)
    mutual_information.to_csv(
        args.output / "03_mutual_information_scores.csv",
        header=True,
    )

    selected_features, decisions = select_nonredundant_features(
        DEFAULT_FEATURES,
        corr_all,
        high_pairs,
        mutual_information,
    )
    decisions.to_csv(
        args.output / "04_feature_removal_decisions.csv",
        index=False,
    )

    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        args.output / "05_selected_features.csv",
        index=False,
    )

    corr_selected = clean_df[selected_features].corr(method="pearson")
    corr_selected.to_csv(args.output / "correlation_matrix_selected.csv")
    save_correlation_heatmap(
        corr_selected,
        "Pearson correlation matrix - selected nonredundant features",
        args.output / "06_correlation_matrix_selected.png",
    )

    # 4. Rank features for pairwise scatter plots
    plot_df = stratified_sample(
        clean_df,
        max_per_class=args.plot_per_class,
    )
    fisher_scores = calculate_fisher_scores(plot_df, DEFAULT_FEATURES)
    fisher_scores.to_csv(
        args.output / "07_fisher_scores.csv",
        header=True,
    )

    top_features = fisher_scores.index[: min(5, len(fisher_scores))].tolist()
    candidate_pairs = list(itertools.combinations(top_features, 2))
    pair_scores = [
        (left, right, fisher_scores[left] + fisher_scores[right])
        for left, right in candidate_pairs
    ]
    pair_scores.sort(key=lambda item: item[2], reverse=True)

    for index, (left, right, _) in enumerate(
        pair_scores[: args.top_pairs],
        start=1,
    ):
        save_single_scatter(
            plot_df,
            left,
            right,
            scatter_dir / f"{index:02d}_{left}_vs_{right}.png",
        )

    # 5. PCA scatter: all 12 features
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(plot_df[DEFAULT_FEATURES])
    pca_all = PCA(n_components=2, random_state=42)
    pca_all_coords = pca_all.fit_transform(X_all_scaled)
    save_projection_scatter(
        pca_all_coords,
        plot_df[LABEL_COLUMN],
        "PC1",
        "PC2",
        (
            "PCA projection - all 12 features "
            f"({pca_all.explained_variance_ratio_.sum():.1%} variance)"
        ),
        args.output / "08_pca_all_12_features.png",
    )

    # 6. PCA scatter: selected features
    scaler_selected = StandardScaler()
    X_selected_scaled = scaler_selected.fit_transform(
        plot_df[selected_features]
    )
    pca_selected = PCA(n_components=2, random_state=42)
    pca_selected_coords = pca_selected.fit_transform(X_selected_scaled)
    save_projection_scatter(
        pca_selected_coords,
        plot_df[LABEL_COLUMN],
        "PC1",
        "PC2",
        (
            "PCA projection - selected features "
            f"({pca_selected.explained_variance_ratio_.sum():.1%} variance)"
        ),
        args.output / "09_pca_selected_features.png",
    )

    # 7. LDA scatter: selected features
    label_encoder = LabelEncoder()
    y_plot = label_encoder.fit_transform(plot_df[LABEL_COLUMN].astype(str))
    max_lda_components = min(
        len(selected_features),
        len(label_encoder.classes_) - 1,
    )
    if max_lda_components >= 2:
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda_coords = lda.fit_transform(X_selected_scaled, y_plot)
        save_projection_scatter(
            lda_coords,
            plot_df[LABEL_COLUMN],
            "LD1",
            "LD2",
            "LDA projection - selected features",
            args.output / "10_lda_selected_features.png",
        )

    # 8. Quantitative separability evaluation
    eval_df = stratified_sample(
        clean_df,
        max_per_class=args.eval_per_class,
    )
    metrics_all = evaluate_separability(eval_df, DEFAULT_FEATURES)
    metrics_selected = evaluate_separability(eval_df, selected_features)

    summary_lines = [
        "PHASE 2 FEATURE ANALYSIS SUMMARY",
        "=" * 42,
        f"Correlation threshold: |r| >= {args.threshold:.2f}",
        f"Rows used for feature selection: {len(clean_df):,}",
        f"Rows used for separability evaluation: {len(eval_df):,}",
        "",
        f"Original features ({len(DEFAULT_FEATURES)}):",
        ", ".join(DEFAULT_FEATURES),
        "",
        f"Selected features ({len(selected_features)}):",
        ", ".join(selected_features),
        "",
        "Dropped features:",
        (
            ", ".join(decisions["dropped_feature"].tolist())
            if not decisions.empty
            else "None"
        ),
        "",
        "Separability metrics - all 12 features:",
        *[f"{key}: {value:.4f}" for key, value in metrics_all.items()],
        "",
        "Separability metrics - selected features:",
        *[f"{key}: {value:.4f}" for key, value in metrics_selected.items()],
        "",
        "Interpretation guide:",
        "- Correlation |r| >= 0.90 usually indicates strong redundancy.",
        "- Silhouette near 1: well separated; near 0: overlapping; below 0: poor separation.",
        "- Macro-F1 near 1: strong multiclass separability.",
        "- PCA is unsupervised; class overlap in PCA does not prove the features are useless.",
        "- LDA is supervised and shows the best linear class separation available in 2D.",
    ]
    (args.output / "11_analysis_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nAnalysis completed.")
    print(f"Selected features: {selected_features}")
    print(f"Outputs saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
