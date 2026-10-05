import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from boundary_retrieval.evaluation import (
    bootstrap_mean_overlap_efs_reduction,
    bootstrap_mean_rr_difference,
    bootstrap_mean_rr_difference_by_document,
    build_fragmentation_pairs,
    build_full_index_fragmentation_pairs,
    build_overlap_fragmentation_comparison,
    build_within_query_spearman,
    summarize_full_index_fragmentation,
    summarize_overlap_fragmentation,
    summarize_within_query_spearman,
)

RETRIEVERS = ("bm25", "e5")
OVERLAP_BASELINE_PHASES = (0, 50, 100, 150)


def load_results_csv(path: str | Path) -> list[dict[str, str]]:
    """Load a saved experiment CSV without changing its recorded values."""
    path = Path(path)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"results file is empty: {path}")

    required = {"query_id", "retriever", "phase", "efs", "rr"}
    missing = required - set(rows[0])

    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"results file is missing required columns: {missing_columns}")

    return rows


def _summarize_controlled_pairs(
    pairs: list[dict[str, Any]], retriever: str
) -> dict[str, Any]:
    matches = [pair for pair in pairs if pair["retriever"] == retriever]

    if not matches:
        raise ValueError(f"no controlled pairs found for retriever: {retriever}")

    diffs = [float(pair["rr_difference"]) for pair in matches]
    lower = 0
    higher = 0
    unchanged = 0

    for diff in diffs:
        if math.isclose(diff, 0.0, abs_tol=1e-12):
            unchanged += 1
        elif diff < 0.0:
            lower += 1
        else:
            higher += 1

    return {
        "n_queries": len(matches),
        "mean_low_efs_rr": mean(float(pair["low_rr"]) for pair in matches),
        "mean_high_efs_rr": mean(float(pair["high_rr"]) for pair in matches),
        "mean_rr_difference": mean(diffs),
        "direction_counts": {
            "lower_at_high_efs": lower,
            "unchanged": unchanged,
            "higher_at_high_efs": higher,
        },
    }


def _summarize_spearman_signs(
    corrs: list[dict[str, Any]], retriever: str
) -> dict[str, int]:
    values = [
        float(row["spearman_rho"])
        for row in corrs
        if row["retriever"] == retriever and row["spearman_rho"] is not None
    ]

    negative = 0
    positive = 0
    zero = 0

    for value in values:
        if math.isclose(value, 0.0, abs_tol=1e-12):
            zero += 1
        elif value < 0.0:
            negative += 1
        else:
            positive += 1

    return {"negative": negative, "zero": zero, "positive": positive}


def _count_overlap_efs_variation(
    overlap_results: list[dict[str, Any]],
) -> dict[str, int]:
    by_query: dict[str, list[float]] = {}

    for row in overlap_results:
        by_query.setdefault(row["query_id"], []).append(float(row["efs"]))

    variable = 0

    for values in by_query.values():
        if not math.isclose(min(values), max(values), abs_tol=1e-12):
            variable += 1

    return {
        "n_queries": len(by_query),
        "n_variable_efs": variable,
        "n_constant_efs": len(by_query) - variable,
    }


def build_analysis_summary(
    primary_results: list[dict[str, Any]],
    overlap_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the paper-facing secondary summaries from saved result rows."""
    controlled_pairs = build_fragmentation_pairs(primary_results)
    corrs = build_within_query_spearman(primary_results)
    full_pairs = build_full_index_fragmentation_pairs(primary_results)

    baseline_overlap_results = [
        row for row in primary_results if int(row["phase"]) in OVERLAP_BASELINE_PHASES
    ]
    overlap_comparisons = build_overlap_fragmentation_comparison(
        baseline_overlap_results, overlap_results
    )

    controlled: dict[str, Any] = {}
    spearman: dict[str, Any] = {}
    full_index: dict[str, Any] = {}

    for retriever in RETRIEVERS:
        controlled[retriever] = {
            **_summarize_controlled_pairs(controlled_pairs, retriever),
            "query_bootstrap": bootstrap_mean_rr_difference(
                controlled_pairs, retriever
            ),
            "document_cluster_bootstrap": bootstrap_mean_rr_difference_by_document(
                primary_results, retriever
            ),
        }

        spearman[retriever] = {
            **summarize_within_query_spearman(corrs, retriever),
            "sign_counts": _summarize_spearman_signs(corrs, retriever),
        }

        full_index[retriever] = summarize_full_index_fragmentation(
            full_pairs, retriever
        )

    return {
        "controlled_probe": controlled,
        "within_query_spearman": spearman,
        "full_index": full_index,
        "overlap_fragmentation": {
            "summary": summarize_overlap_fragmentation(overlap_comparisons),
            "query_bootstrap": bootstrap_mean_overlap_efs_reduction(
                overlap_comparisons
            ),
            "within_query_variation": _count_overlap_efs_variation(overlap_results),
        },
    }


def save_analysis_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Save derived numerical summaries as formatted JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return output_path


def write_analysis_artifacts(
    data_dir: str | Path,
    primary_results_path: str | Path,
    overlap_results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Regenerate summary JSON, qualitative cases, and the main figure."""
    primary_results = load_results_csv(primary_results_path)
    overlap_results = load_results_csv(overlap_results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = save_analysis_summary(
        build_analysis_summary(primary_results, overlap_results),
        output_dir / "analysis_summary.json",
    )

    # These imports are only needed when regenerating the full set of artifacts.
    from boundary_retrieval.experiment import (
        build_error_case_records,
        gen_efs_rr_figure,
        load_validation_examples,
        save_error_case_records,
    )

    val_examples = load_validation_examples(data_dir)
    error_records = build_error_case_records(primary_results, val_examples)
    error_path = save_error_case_records(error_records, output_dir / "error_cases.json")
    figure_path = gen_efs_rr_figure(
        primary_results, output_dir / "figures" / "efs_vs_rr.png"
    )

    return {
        "summary": summary_path,
        "error_cases": error_path,
        "figure": figure_path,
    }


def main(argv: list[str] | None = None) -> int:
    """Regenerate derived analysis artifacts from saved experiment CSVs."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate numerical summaries, qualitative error cases, and the "
            "EFS-versus-RR figure from saved experiment results."
        )
    )
    parser.add_argument(
        "data_dir", type=Path, help="Directory containing the Doc2Dial JSON files."
    )
    parser.add_argument("primary_results", type=Path, help="Saved primary-results CSV.")
    parser.add_argument(
        "overlap_results", type=Path, help="Saved overlap-ablation CSV."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory for regenerated analysis artifacts."
    )

    args = parser.parse_args(argv)
    paths = write_analysis_artifacts(
        args.data_dir,
        args.primary_results,
        args.overlap_results,
        args.output_dir,
    )

    for label, path in paths.items():
        print(f"Saved {label}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
