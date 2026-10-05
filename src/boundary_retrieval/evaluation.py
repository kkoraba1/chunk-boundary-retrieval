import math
import re
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
from document_chunking_utility.chunkers import TextChunk
from scipy.stats import spearmanr

from boundary_retrieval.data import get_evidence_word_bounds


def _chunk_word_bounds(doc_text: str, chunk: TextChunk) -> tuple[int, int]:
    """Return a chunk's word range, with an exclusive end."""
    words = list(re.finditer(r"\S+", doc_text))
    chunk_words = [
        idx
        for idx, match in enumerate(words)
        if match.start() < chunk.end_char and match.end() > chunk.start_char
    ]

    if not chunk_words:
        raise ValueError("Chunk does not contain any source words")

    return chunk_words[0], chunk_words[-1] + 1


def evidence_overlap_fraction(example: dict[str, Any], chunk: TextChunk) -> float:
    """Return the fraction of gold evidence contained in one chunk."""
    evidence_start, evidence_end = get_evidence_word_bounds(example)
    chunk_start, chunk_end = _chunk_word_bounds(example["document_text"], chunk)

    overlap_start = max(evidence_start, chunk_start)
    overlap_end = min(evidence_end, chunk_end)
    overlap_words = max(0, overlap_end - overlap_start)
    evidence_words = evidence_end - evidence_start

    return overlap_words / evidence_words


def best_evidence_chunk_index(example: dict[str, Any], chunks: list[TextChunk]) -> int:
    """Return the list index of the chunk containing the most evidence."""
    if not chunks:
        raise ValueError("chunks must not be empty")

    best_idx = 0
    best_frac = evidence_overlap_fraction(example, chunks[0])

    for idx, chunk in enumerate(chunks[1:], start=1):
        fraction = evidence_overlap_fraction(example, chunk)

        if fraction > best_frac:
            best_idx = idx
            best_frac = fraction

    return best_idx


def evidence_fragmentation_score(
    example: dict[str, Any], chunks: list[TextChunk]
) -> float:
    """Compute the Evidence Fragmentation Score for one segmentation."""
    best_idx = best_evidence_chunk_index(example, chunks)
    best_frac = evidence_overlap_fraction(example, chunks[best_idx])

    return 1.0 - best_frac


def reciprocal_rank(
    ranked_passages: list[tuple[int, float]], target_index: int
) -> float:
    """Return the target's reciprocal rank, averaging ranks for exact score ties."""
    target_score = None

    for passage_idx, score in ranked_passages:
        if passage_idx == target_index:
            target_score = float(score)
            break

    if target_score is None:
        return 0.0

    higher_count = sum(
        1
        for passage_idx, score in ranked_passages
        if passage_idx != target_index and float(score) > target_score
    )
    tied_count = sum(
        1
        for passage_idx, score in ranked_passages
        if passage_idx != target_index and float(score) == target_score
    )

    avg_rank = higher_count + 1.0 + (tied_count / 2.0)
    return 1.0 / avg_rank


def graded_evidence_relevance(
    example: dict[str, Any], chunks: list[TextChunk]
) -> list[float]:
    """Return the fraction of gold evidence contained in each chunk."""
    return [evidence_overlap_fraction(example, chunk) for chunk in chunks]


def graded_ndcg_at_5(
    ranked_passages: list[tuple[int, float]], rel: list[float]
) -> float:
    """Return graded nDCG@5 for a passage ranking."""
    dcg = 0.0

    for rank, (passage_idx, _) in enumerate(ranked_passages[:5], start=1):
        gain = (2 ** rel[passage_idx]) - 1
        dcg += gain / math.log2(rank + 1)

    ideal_rel = sorted(rel, reverse=True)[:5]
    ideal_dcg = 0.0

    for rank, grade in enumerate(ideal_rel, start=1):
        gain = (2**grade) - 1
        ideal_dcg += gain / math.log2(rank + 1)

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg


def _evidence_overlap_interval(
    example: dict[str, Any], chunk: TextChunk
) -> tuple[int, int] | None:
    """Return the source-word interval where a chunk overlaps the evidence."""
    evidence_start, evidence_end = get_evidence_word_bounds(example)
    chunk_start, chunk_end = _chunk_word_bounds(example["document_text"], chunk)

    overlap_start = max(evidence_start, chunk_start)
    overlap_end = min(evidence_end, chunk_end)

    if overlap_start >= overlap_end:
        return None

    return overlap_start, overlap_end


def _covered_word_count(intervals: list[tuple[int, int]]) -> int:
    """Return the number of unique source words covered by intervals."""
    if not intervals:
        return 0

    ordered = sorted(intervals)
    current_start, current_end = ordered[0]
    covered = 0

    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end

    covered += current_end - current_start
    return covered


def oracle_normalized_evidence_coverage_at_5(
    example: dict[str, Any],
    chunks: list[TextChunk],
    ranked_passages: list[tuple[int, float]],
) -> float:
    """Divide top-5 evidence coverage by the best possible top-5 coverage."""
    overlap_intervals = [_evidence_overlap_interval(example, chunk) for chunk in chunks]

    retrieved_intervals = [
        overlap_intervals[passage_idx]
        for passage_idx, _ in ranked_passages[:5]
        if overlap_intervals[passage_idx] is not None
    ]
    actual_coverage = _covered_word_count(retrieved_intervals)

    relevant_intervals = [
        interval for interval in overlap_intervals if interval is not None
    ]

    if not relevant_intervals:
        return 0.0

    oracle_size = min(5, len(relevant_intervals))
    oracle_coverage = max(
        _covered_word_count(list(selected))
        for selected in combinations(relevant_intervals, oracle_size)
    )

    if oracle_coverage == 0:
        return 0.0

    return actual_coverage / oracle_coverage


def build_fragmentation_pairs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build low/high-EFS RR pairs for each query and retriever.

    Average RR across phases tied for the lowest or highest EFS. Skip queries
    whose EFS does not change. Each difference is high-EFS RR minus low-EFS RR.
    """
    grouped_results: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in results:
        key = (row["query_id"], row["retriever"])
        grouped_results.setdefault(key, []).append(row)

    pairs = []

    for query_id, retriever in sorted(grouped_results):
        group = grouped_results[(query_id, retriever)]
        efs_values = [float(row["efs"]) for row in group]
        low_efs = min(efs_values)
        high_efs = max(efs_values)

        if math.isclose(low_efs, high_efs, abs_tol=1e-12):
            continue

        low_rows = [row for row in group if math.isclose(float(row["efs"]), low_efs)]
        high_rows = [row for row in group if math.isclose(float(row["efs"]), high_efs)]

        low_rr = sum(float(row["rr"]) for row in low_rows) / len(low_rows)
        high_rr = sum(float(row["rr"]) for row in high_rows) / len(high_rows)

        pairs.append(
            {
                "query_id": query_id,
                "retriever": retriever,
                "low_efs": low_efs,
                "high_efs": high_efs,
                "low_rr": low_rr,
                "high_rr": high_rr,
                "rr_difference": high_rr - low_rr,
            }
        )

    return pairs


def bootstrap_mean_rr_difference(
    pairs: list[dict[str, Any]],
    retriever: str,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the mean paired RR difference for one retriever."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    diffs = np.array(
        [
            float(pair["rr_difference"])
            for pair in pairs
            if pair["retriever"] == retriever
        ],
        dtype=float,
    )

    if len(diffs) == 0:
        raise ValueError(f"no pairs found for retriever: {retriever}")

    rng = np.random.default_rng(seed)
    sample_idxs = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    boot_means = diffs[sample_idxs].mean(axis=1)

    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])

    return {
        "retriever": retriever,
        "n_queries": len(diffs),
        "mean_rr_difference": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def bootstrap_mean_rr_difference_by_document(
    results: list[dict[str, Any]],
    retriever: str,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap mean paired RR difference by resampling source documents."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    query_docs: dict[str, str] = {}

    for row in results:
        query_id = row["query_id"]
        doc_id = row["document_id"]

        if query_id in query_docs and query_docs[query_id] != doc_id:
            raise ValueError(f"query {query_id} maps to multiple documents")

        query_docs[query_id] = doc_id

    pairs = [
        pair
        for pair in build_fragmentation_pairs(results)
        if pair["retriever"] == retriever
    ]

    if not pairs:
        raise ValueError(f"no pairs found for retriever: {retriever}")

    doc_diffs: dict[str, list[float]] = {}

    for pair in pairs:
        query_id = pair["query_id"]

        if query_id not in query_docs:
            raise ValueError(f"missing document for query: {query_id}")

        doc_id = query_docs[query_id]
        doc_diffs.setdefault(doc_id, []).append(float(pair["rr_difference"]))

    doc_ids = sorted(doc_diffs)

    doc_sums = np.array([sum(doc_diffs[doc_id]) for doc_id in doc_ids], dtype=float)
    doc_counts = np.array([len(doc_diffs[doc_id]) for doc_id in doc_ids], dtype=float)

    rng = np.random.default_rng(seed)
    sample_idxs = rng.integers(0, len(doc_ids), size=(n_resamples, len(doc_ids)))

    boot_sums = doc_sums[sample_idxs].sum(axis=1)
    boot_counts = doc_counts[sample_idxs].sum(axis=1)
    boot_means = boot_sums / boot_counts

    diffs = np.array([float(pair["rr_difference"]) for pair in pairs], dtype=float)

    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])

    return {
        "retriever": retriever,
        "n_queries": len(diffs),
        "n_documents": len(doc_ids),
        "mean_rr_difference": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def build_within_query_spearman(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute EFS/RR Spearman correlation within each query."""
    grouped_results: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in results:
        key = (row["query_id"], row["retriever"])
        grouped_results.setdefault(key, []).append(row)

    corrs = []

    for query_id, retriever in sorted(grouped_results):
        group = grouped_results[(query_id, retriever)]
        efs_values = [float(row["efs"]) for row in group]
        rr_values = [float(row["rr"]) for row in group]

        if len(set(efs_values)) < 2 or len(set(rr_values)) < 2:
            rho = None
        else:
            rho = float(spearmanr(efs_values, rr_values).statistic)

        corrs.append(
            {
                "query_id": query_id,
                "retriever": retriever,
                "n_phases": len(group),
                "spearman_rho": rho,
            }
        )

    return corrs


def summarize_within_query_spearman(
    corrs: list[dict[str, Any]], retriever: str
) -> dict[str, Any]:
    """Summarize defined within-query Spearman correlations."""
    matches = [row for row in corrs if row["retriever"] == retriever]

    if not matches:
        raise ValueError(f"no correlations found for retriever: {retriever}")

    defined_rhos = [
        float(row["spearman_rho"]) for row in matches if row["spearman_rho"] is not None
    ]

    mean_rho = None
    median_rho = None

    if defined_rhos:
        mean_rho = float(np.mean(defined_rhos))
        median_rho = float(np.median(defined_rhos))

    return {
        "retriever": retriever,
        "n_queries": len(matches),
        "n_defined": len(defined_rhos),
        "n_undefined": len(matches) - len(defined_rhos),
        "mean_rho": mean_rho,
        "median_rho": median_rho,
    }


def build_efs_rr_plot_data(
    results: list[dict[str, Any]], retriever: str
) -> list[dict[str, Any]]:
    """Count repeated EFS/RR points for one retriever's plot."""
    points = Counter(
        (float(row["efs"]), float(row["rr"]))
        for row in results
        if row["retriever"] == retriever
    )

    if not points:
        raise ValueError(f"no results found for retriever: {retriever}")

    return [
        {"efs": efs, "rr": rr, "count": count}
        for (efs, rr), count in sorted(points.items())
    ]


def build_full_index_fragmentation_pairs(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build low/high-EFS pairs for the full-index validation metrics."""
    grouped_results: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in results:
        key = (row["query_id"], row["retriever"])
        grouped_results.setdefault(key, []).append(row)

    pairs = []

    for query_id, retriever in sorted(grouped_results):
        group = grouped_results[(query_id, retriever)]
        efs_values = [float(row["efs"]) for row in group]
        low_efs = min(efs_values)
        high_efs = max(efs_values)

        if math.isclose(low_efs, high_efs, abs_tol=1e-12):
            continue

        low_rows = [row for row in group if math.isclose(float(row["efs"]), low_efs)]
        high_rows = [row for row in group if math.isclose(float(row["efs"]), high_efs)]

        low_ndcg = sum(float(row["ndcg_at_5"]) for row in low_rows) / len(low_rows)
        high_ndcg = sum(float(row["ndcg_at_5"]) for row in high_rows) / len(high_rows)

        low_coverage = sum(
            float(row["oracle_normalized_coverage_at_5"]) for row in low_rows
        ) / len(low_rows)
        high_coverage = sum(
            float(row["oracle_normalized_coverage_at_5"]) for row in high_rows
        ) / len(high_rows)

        pairs.append(
            {
                "query_id": query_id,
                "retriever": retriever,
                "low_efs": low_efs,
                "high_efs": high_efs,
                "low_ndcg_at_5": low_ndcg,
                "high_ndcg_at_5": high_ndcg,
                "ndcg_at_5_difference": high_ndcg - low_ndcg,
                "low_oracle_normalized_coverage_at_5": low_coverage,
                "high_oracle_normalized_coverage_at_5": high_coverage,
                "oracle_normalized_coverage_at_5_difference": high_coverage
                - low_coverage,
            }
        )

    return pairs


def summarize_full_index_fragmentation(
    pairs: list[dict[str, Any]], retriever: str
) -> dict[str, Any]:
    """Summarize paired full-index validation metrics for one retriever."""
    matches = [pair for pair in pairs if pair["retriever"] == retriever]

    if not matches:
        raise ValueError(f"no full-index pairs found for retriever: {retriever}")

    def avg(field: str) -> float:
        return float(np.mean([float(pair[field]) for pair in matches]))

    return {
        "retriever": retriever,
        "n_queries": len(matches),
        "mean_low_ndcg_at_5": avg("low_ndcg_at_5"),
        "mean_high_ndcg_at_5": avg("high_ndcg_at_5"),
        "mean_ndcg_at_5_difference": avg("ndcg_at_5_difference"),
        "mean_low_oracle_normalized_coverage_at_5": avg(
            "low_oracle_normalized_coverage_at_5"
        ),
        "mean_high_oracle_normalized_coverage_at_5": avg(
            "high_oracle_normalized_coverage_at_5"
        ),
        "mean_oracle_normalized_coverage_at_5_difference": avg(
            "oracle_normalized_coverage_at_5_difference"
        ),
    }


def select_error_cases(
    results: list[dict[str, Any]], n_cases: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Pick examples of failures, robust results, and retriever disagreements.

    Failures have an RR drop for at least one retriever. Robust cases have no
    drop. Disagreements have a drop for one retriever but not the other.
    """
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")

    grouped_results: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in results:
        key = (row["query_id"], row["retriever"])
        grouped_results.setdefault(key, []).append(row)

    retriever_summaries: dict[str, dict[str, dict[str, Any]]] = {}

    for (query_id, retriever), group in grouped_results.items():
        efs_values = [float(row["efs"]) for row in group]
        low_efs = min(efs_values)
        high_efs = max(efs_values)

        low_rows = [row for row in group if math.isclose(float(row["efs"]), low_efs)]
        high_rows = [row for row in group if math.isclose(float(row["efs"]), high_efs)]

        low_rr = sum(float(row["rr"]) for row in low_rows) / len(low_rows)
        high_rr = sum(float(row["rr"]) for row in high_rows) / len(high_rows)

        retriever_summaries.setdefault(query_id, {})[retriever] = {
            "document_id": group[0]["document_id"],
            "low_efs": low_efs,
            "high_efs": high_efs,
            "low_phases": tuple(sorted(int(row["phase"]) for row in low_rows)),
            "high_phases": tuple(sorted(int(row["phase"]) for row in high_rows)),
            "low_rr": low_rr,
            "high_rr": high_rr,
            "rr_difference": high_rr - low_rr,
        }

    query_summaries = []

    for query_id in sorted(retriever_summaries):
        by_retriever = retriever_summaries[query_id]

        if "bm25" not in by_retriever or "e5" not in by_retriever:
            continue

        bm25 = by_retriever["bm25"]
        e5 = by_retriever["e5"]

        query_summaries.append(
            {
                "query_id": query_id,
                "document_id": bm25["document_id"],
                "low_efs": bm25["low_efs"],
                "high_efs": bm25["high_efs"],
                "low_phases": bm25["low_phases"],
                "high_phases": bm25["high_phases"],
                "bm25_low_rr": bm25["low_rr"],
                "bm25_high_rr": bm25["high_rr"],
                "bm25_rr_difference": bm25["rr_difference"],
                "e5_low_rr": e5["low_rr"],
                "e5_high_rr": e5["high_rr"],
                "e5_rr_difference": e5["rr_difference"],
            }
        )

    failures = []
    robust = []
    disagreements = []

    for summary in query_summaries:
        if summary["high_efs"] <= summary["low_efs"]:
            continue

        bm25_diff = summary["bm25_rr_difference"]
        e5_diff = summary["e5_rr_difference"]

        if min(bm25_diff, e5_diff) < 0.0:
            failure = dict(summary)
            failure["focus_retriever"] = "bm25" if bm25_diff <= e5_diff else "e5"
            failures.append(failure)

        if bm25_diff >= 0.0 and e5_diff >= 0.0:
            robust.append(dict(summary))

        retrievers_disagree = bm25_diff < 0.0 <= e5_diff or e5_diff < 0.0 <= bm25_diff

        if retrievers_disagree:
            disagreement = dict(summary)
            disagreement["rr_difference_gap"] = abs(bm25_diff - e5_diff)
            disagreements.append(disagreement)

    failures.sort(
        key=lambda row: (
            min(row["bm25_rr_difference"], row["e5_rr_difference"]),
            -row["high_efs"],
            row["query_id"],
        )
    )
    robust.sort(
        key=lambda row: (
            -min(row["bm25_high_rr"], row["e5_high_rr"]),
            abs(row["bm25_rr_difference"]) + abs(row["e5_rr_difference"]),
            -row["high_efs"],
            row["query_id"],
        )
    )
    disagreements.sort(
        key=lambda row: (-row["rr_difference_gap"], -row["high_efs"], row["query_id"])
    )

    return {
        "failures": failures[:n_cases],
        "robust": robust[:n_cases],
        "disagreements": disagreements[:n_cases],
    }


def _build_query_phase_efs(
    results: list[dict[str, Any]],
) -> dict[str, dict[int, float]]:
    """Collect one EFS value per query and phase."""
    query_phases: dict[str, dict[int, float]] = {}

    for row in results:
        query_id = row["query_id"]
        phase = int(row["phase"])
        efs = float(row["efs"])
        phases = query_phases.setdefault(query_id, {})

        if phase in phases and not math.isclose(phases[phase], efs):
            raise ValueError(f"inconsistent EFS for query {query_id} at phase {phase}")

        phases[phase] = efs

    return query_phases


def build_overlap_fragmentation_comparison(
    baseline_results: list[dict[str, Any]], overlap_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare each query's EFS across the reduced baseline and overlap cycles."""
    base = _build_query_phase_efs(baseline_results)
    overlap = _build_query_phase_efs(overlap_results)

    if set(base) != set(overlap):
        raise ValueError("baseline and overlap query sets do not match")

    comparisons = []

    for query_id in sorted(base):
        base_values = list(base[query_id].values())
        overlap_values = list(overlap[query_id].values())

        if len(base_values) != len(overlap_values):
            raise ValueError(
                f"baseline and overlap phase counts do not match for {query_id}"
            )

        if not base_values:
            raise ValueError(f"query has no phase results: {query_id}")

        base_mean = float(np.mean(base_values))
        overlap_mean = float(np.mean(overlap_values))
        base_max = max(base_values)
        overlap_max = max(overlap_values)

        comparisons.append(
            {
                "query_id": query_id,
                "n_phases": len(base_values),
                "baseline_mean_efs": base_mean,
                "overlap_mean_efs": overlap_mean,
                "mean_efs_reduction": base_mean - overlap_mean,
                "baseline_max_efs": base_max,
                "overlap_max_efs": overlap_max,
                "max_efs_reduction": base_max - overlap_max,
            }
        )

    return comparisons


def summarize_overlap_fragmentation(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize how overlap changes query-level fragmentation."""
    if not comparisons:
        raise ValueError("no overlap fragmentation comparisons found")

    reductions = [float(row["mean_efs_reduction"]) for row in comparisons]

    reduced = sum(
        value > 0.0 and not math.isclose(value, 0.0, abs_tol=1e-12)
        for value in reductions
    )
    increased = sum(
        value < 0.0 and not math.isclose(value, 0.0, abs_tol=1e-12)
        for value in reductions
    )
    unchanged = len(reductions) - reduced - increased

    def avg(field: str) -> float:
        return float(np.mean([float(row[field]) for row in comparisons]))

    return {
        "n_queries": len(comparisons),
        "mean_baseline_mean_efs": avg("baseline_mean_efs"),
        "mean_overlap_mean_efs": avg("overlap_mean_efs"),
        "mean_efs_reduction": avg("mean_efs_reduction"),
        "mean_baseline_max_efs": avg("baseline_max_efs"),
        "mean_overlap_max_efs": avg("overlap_max_efs"),
        "mean_max_efs_reduction": avg("max_efs_reduction"),
        "n_queries_reduced": reduced,
        "n_queries_unchanged": unchanged,
        "n_queries_increased": increased,
    }


def bootstrap_mean_overlap_efs_reduction(
    comparisons: list[dict[str, Any]],
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the mean query-level EFS reduction from overlap."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    reductions = np.array(
        [float(row["mean_efs_reduction"]) for row in comparisons], dtype=float
    )

    if len(reductions) == 0:
        raise ValueError("no overlap fragmentation comparisons found")

    rng = np.random.default_rng(seed)
    sample_idxs = rng.integers(0, len(reductions), size=(n_resamples, len(reductions)))
    boot_means = reductions[sample_idxs].mean(axis=1)

    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])

    return {
        "n_queries": len(reductions),
        "mean_efs_reduction": float(reductions.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def build_overlap_retrieval_comparison(
    baseline_results: list[dict[str, Any]], overlap_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare retrieval changes where both runs have a low/high-EFS pair."""
    base_result_keys = {(row["query_id"], row["retriever"]) for row in baseline_results}
    overlap_result_keys = {
        (row["query_id"], row["retriever"]) for row in overlap_results
    }

    if base_result_keys != overlap_result_keys:
        raise ValueError("baseline and overlap retrieval query sets do not match")

    base_rr = {
        (row["query_id"], row["retriever"]): row
        for row in build_fragmentation_pairs(baseline_results)
    }
    overlap_rr = {
        (row["query_id"], row["retriever"]): row
        for row in build_fragmentation_pairs(overlap_results)
    }
    base_full = {
        (row["query_id"], row["retriever"]): row
        for row in build_full_index_fragmentation_pairs(baseline_results)
    }
    overlap_full = {
        (row["query_id"], row["retriever"]): row
        for row in build_full_index_fragmentation_pairs(overlap_results)
    }

    if set(base_rr) != set(base_full):
        raise ValueError("baseline retrieval contrast sets do not match")

    if set(overlap_rr) != set(overlap_full):
        raise ValueError("overlap retrieval contrast sets do not match")

    shared_keys = set(base_rr) & set(overlap_rr)
    comparisons = []

    for query_id, retriever in sorted(shared_keys):
        base_rr_row = base_rr[(query_id, retriever)]
        overlap_rr_row = overlap_rr[(query_id, retriever)]
        base_full_row = base_full[(query_id, retriever)]
        overlap_full_row = overlap_full[(query_id, retriever)]

        base_rr_diff = float(base_rr_row["rr_difference"])
        overlap_rr_diff = float(overlap_rr_row["rr_difference"])
        base_ndcg_diff = float(base_full_row["ndcg_at_5_difference"])
        overlap_ndcg_diff = float(overlap_full_row["ndcg_at_5_difference"])
        base_coverage_diff = float(
            base_full_row["oracle_normalized_coverage_at_5_difference"]
        )
        overlap_coverage_diff = float(
            overlap_full_row["oracle_normalized_coverage_at_5_difference"]
        )

        comparisons.append(
            {
                "query_id": query_id,
                "retriever": retriever,
                "baseline_rr_difference": base_rr_diff,
                "overlap_rr_difference": overlap_rr_diff,
                "rr_mitigation": overlap_rr_diff - base_rr_diff,
                "baseline_ndcg_at_5_difference": base_ndcg_diff,
                "overlap_ndcg_at_5_difference": overlap_ndcg_diff,
                "ndcg_at_5_mitigation": overlap_ndcg_diff - base_ndcg_diff,
                "baseline_oracle_normalized_coverage_at_5_difference": base_coverage_diff,
                "overlap_oracle_normalized_coverage_at_5_difference": overlap_coverage_diff,
                "oracle_normalized_coverage_at_5_mitigation": (
                    overlap_coverage_diff - base_coverage_diff
                ),
            }
        )

    return comparisons


def summarize_overlap_retrieval(
    comparisons: list[dict[str, Any]], retriever: str
) -> dict[str, Any]:
    """Summarize how overlap changes the fragmentation effect for one retriever."""
    matches = [row for row in comparisons if row["retriever"] == retriever]

    if not matches:
        raise ValueError(
            f"no overlap retrieval comparisons found for retriever: {retriever}"
        )

    def avg(field: str) -> float:
        return float(np.mean([float(row[field]) for row in matches]))

    return {
        "retriever": retriever,
        "n_queries": len(matches),
        "mean_baseline_rr_difference": avg("baseline_rr_difference"),
        "mean_overlap_rr_difference": avg("overlap_rr_difference"),
        "mean_rr_mitigation": avg("rr_mitigation"),
        "mean_baseline_ndcg_at_5_difference": avg("baseline_ndcg_at_5_difference"),
        "mean_overlap_ndcg_at_5_difference": avg("overlap_ndcg_at_5_difference"),
        "mean_ndcg_at_5_mitigation": avg("ndcg_at_5_mitigation"),
        "mean_baseline_oracle_normalized_coverage_at_5_difference": avg(
            "baseline_oracle_normalized_coverage_at_5_difference"
        ),
        "mean_overlap_oracle_normalized_coverage_at_5_difference": avg(
            "overlap_oracle_normalized_coverage_at_5_difference"
        ),
        "mean_oracle_normalized_coverage_at_5_mitigation": avg(
            "oracle_normalized_coverage_at_5_mitigation"
        ),
    }


def bootstrap_mean_overlap_rr_mitigation(
    comparisons: list[dict[str, Any]],
    retriever: str,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the mean RR mitigation from overlap for one retriever."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    mitigation = np.array(
        [
            float(row["rr_mitigation"])
            for row in comparisons
            if row["retriever"] == retriever
        ],
        dtype=float,
    )

    if len(mitigation) == 0:
        raise ValueError(
            f"no overlap retrieval comparisons found for retriever: {retriever}"
        )

    rng = np.random.default_rng(seed)
    sample_idxs = rng.integers(0, len(mitigation), size=(n_resamples, len(mitigation)))
    boot_means = mitigation[sample_idxs].mean(axis=1)

    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])

    return {
        "retriever": retriever,
        "n_queries": len(mitigation),
        "mean_rr_mitigation": float(mitigation.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "seed": seed,
    }
