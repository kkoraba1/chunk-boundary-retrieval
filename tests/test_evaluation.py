import math

import pytest
from document_chunking_utility.chunkers import TextChunk

from boundary_retrieval.evaluation import (
    best_evidence_chunk_index,
    bootstrap_mean_overlap_efs_reduction,
    bootstrap_mean_overlap_rr_mitigation,
    bootstrap_mean_rr_difference,
    bootstrap_mean_rr_difference_by_document,
    build_efs_rr_plot_data,
    build_fragmentation_pairs,
    build_full_index_fragmentation_pairs,
    build_overlap_fragmentation_comparison,
    build_overlap_retrieval_comparison,
    build_within_query_spearman,
    evidence_fragmentation_score,
    evidence_overlap_fraction,
    graded_evidence_relevance,
    graded_ndcg_at_5,
    oracle_normalized_evidence_coverage_at_5,
    reciprocal_rank,
    select_error_cases,
    summarize_full_index_fragmentation,
    summarize_overlap_fragmentation,
    summarize_overlap_retrieval,
    summarize_within_query_spearman,
)


def make_example(doc_text: str, evidence_text: str) -> dict:
    evidence_start = doc_text.index(evidence_text)
    evidence_end = evidence_start + len(evidence_text)

    return {
        "document_text": doc_text,
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
    }


def make_chunk(doc_text: str, chunk_idx: int, chunk_text: str) -> TextChunk:
    start_char = doc_text.index(chunk_text)
    end_char = start_char + len(chunk_text)

    return TextChunk(
        chunk_index=chunk_idx,
        chunk_text=chunk_text,
        start_char=start_char,
        end_char=end_char,
    )


def test_overlap_fraction_for_fully_contained_evidence():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "gamma delta")
    chunk = make_chunk(doc, 0, "beta gamma delta epsilon")

    assert evidence_overlap_fraction(example, chunk) == 1.0


def test_overlap_fraction_uses_source_words_with_irregular_whitespace():
    doc = "alpha  beta\ngamma\tdelta epsilon zeta"
    example = make_example(doc, "beta\ngamma\tdelta epsilon")
    chunk = make_chunk(doc, 0, "alpha  beta\ngamma")

    assert evidence_overlap_fraction(example, chunk) == 0.5


def test_fragmentation_score_for_partially_split_evidence():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "beta gamma delta")
    chunks = [
        make_chunk(doc, 0, "alpha beta"),
        make_chunk(doc, 1, "gamma delta epsilon zeta"),
    ]

    assert evidence_fragmentation_score(example, chunks) == pytest.approx(1 / 3)


def test_fragmentation_score_for_even_split():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "beta gamma delta epsilon")
    chunks = [
        make_chunk(doc, 0, "alpha beta gamma"),
        make_chunk(doc, 1, "delta epsilon zeta"),
    ]

    assert evidence_fragmentation_score(example, chunks) == 0.5


def test_best_evidence_chunk_index_selects_largest_overlap():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "beta gamma delta")
    chunks = [
        make_chunk(doc, 0, "alpha beta"),
        make_chunk(doc, 1, "gamma delta epsilon zeta"),
    ]

    assert best_evidence_chunk_index(example, chunks) == 1


def test_best_evidence_chunk_index_keeps_first_chunk_on_tie():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "beta gamma delta epsilon")
    chunks = [
        make_chunk(doc, 0, "alpha beta gamma"),
        make_chunk(doc, 1, "delta epsilon zeta"),
    ]

    assert best_evidence_chunk_index(example, chunks) == 0


def test_best_evidence_chunk_index_rejects_empty_chunks():
    doc = "alpha beta gamma"
    example = make_example(doc, "beta")

    with pytest.raises(ValueError, match="chunks must not be empty"):
        best_evidence_chunk_index(example, [])


def test_reciprocal_rank_for_first_result():
    ranked = [(2, 0.9), (0, 0.6), (1, 0.2)]

    assert reciprocal_rank(ranked, 2) == 1.0


def test_reciprocal_rank_for_second_result():
    ranked = [(2, 0.9), (0, 0.6), (1, 0.2)]

    assert reciprocal_rank(ranked, 0) == 0.5


def test_reciprocal_rank_for_third_result():
    ranked = [(2, 0.9), (0, 0.6), (1, 0.2)]

    assert reciprocal_rank(ranked, 1) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_target_is_missing():
    ranked = [(2, 0.9), (0, 0.6)]

    assert reciprocal_rank(ranked, 1) == 0.0


def test_reciprocal_rank_uses_average_rank_for_exact_score_tie():
    ranked = [(0, 0.8), (1, 0.8), (2, 0.4)]

    assert reciprocal_rank(ranked, 0) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, 1) == pytest.approx(2 / 3)


def test_reciprocal_rank_is_neutral_to_target_order_within_tie():
    first = [(0, 0.8), (1, 0.8), (2, 0.4)]
    second = [(1, 0.8), (0, 0.8), (2, 0.4)]

    assert reciprocal_rank(first, 0) == reciprocal_rank(second, 0)


def test_graded_relevance_for_contained_evidence():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "gamma delta")
    chunks = [
        make_chunk(doc, 0, "alpha beta"),
        make_chunk(doc, 1, "gamma delta"),
        make_chunk(doc, 2, "epsilon zeta"),
    ]

    rel = graded_evidence_relevance(example, chunks)

    assert rel == [0.0, 1.0, 0.0]


def test_graded_relevance_for_fragmented_evidence():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "beta gamma delta")
    chunks = [
        make_chunk(doc, 0, "alpha beta"),
        make_chunk(doc, 1, "gamma delta epsilon"),
        make_chunk(doc, 2, "zeta"),
    ]

    rel = graded_evidence_relevance(example, chunks)

    assert rel == pytest.approx([1 / 3, 2 / 3, 0.0])


def test_graded_relevance_is_zero_for_chunks_without_evidence():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "gamma delta")
    chunks = [make_chunk(doc, 0, "alpha beta"), make_chunk(doc, 1, "epsilon zeta")]

    assert graded_evidence_relevance(example, chunks) == [0.0, 0.0]


def test_graded_relevance_accepts_empty_chunk_list():
    doc = "alpha beta gamma"
    example = make_example(doc, "beta")

    assert graded_evidence_relevance(example, []) == []


def test_graded_ndcg_at_5_for_perfect_ranking():
    rel = [1.0, 0.5, 0.0]
    ranked = [(0, 0.9), (1, 0.7), (2, 0.2)]

    assert graded_ndcg_at_5(ranked, rel) == 1.0


def test_graded_ndcg_at_5_for_imperfect_ranking():
    rel = [1.0, 0.5, 0.0]
    ranked = [(2, 0.9), (1, 0.7), (0, 0.2)]

    dcg = ((2**0.5) - 1) / math.log2(3) + ((2**1.0) - 1) / math.log2(4)
    idcg = ((2**1.0) - 1) / math.log2(2) + ((2**0.5) - 1) / math.log2(3)

    assert graded_ndcg_at_5(ranked, rel) == pytest.approx(dcg / idcg)


def test_graded_ndcg_at_5_returns_zero_without_relevant_chunks():
    rel = [0.0, 0.0, 0.0]
    ranked = [(0, 0.9), (1, 0.7), (2, 0.2)]

    assert graded_ndcg_at_5(ranked, rel) == 0.0


def test_graded_ndcg_at_5_ignores_results_after_rank_five():
    rel = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.4)]

    assert graded_ndcg_at_5(ranked, rel) == 0.0


def test_normalized_coverage_is_one_for_oracle_ranking():
    doc = "alpha beta gamma delta epsilon zeta"
    example = make_example(doc, "beta gamma delta epsilon")
    chunks = [
        make_chunk(doc, 0, "alpha beta gamma"),
        make_chunk(doc, 1, "delta epsilon zeta"),
    ]
    ranked = [(0, 0.9), (1, 0.8)]

    assert oracle_normalized_evidence_coverage_at_5(example, chunks, ranked) == 1.0


def test_normalized_coverage_does_not_double_count_overlap():
    doc = "zero one two three four five six seven eight nine ten"
    example = make_example(doc, "one two three four five six seven eight")
    chunks = [
        make_chunk(doc, 0, "zero one two three four"),
        make_chunk(doc, 1, "two three four five"),
        make_chunk(doc, 2, "five six seven eight nine"),
        make_chunk(doc, 3, "zero"),
        make_chunk(doc, 4, "nine"),
        make_chunk(doc, 5, "ten"),
    ]
    ranked = [(0, 0.9), (1, 0.8), (3, 0.7), (4, 0.6), (5, 0.5), (2, 0.4)]

    assert oracle_normalized_evidence_coverage_at_5(
        example, chunks, ranked
    ) == pytest.approx(5 / 8)


def test_normalized_coverage_is_zero_when_top_five_miss_evidence():
    doc = "alpha beta gamma delta epsilon zeta eta theta"
    example = make_example(doc, "gamma delta")
    chunks = [
        make_chunk(doc, 0, "alpha"),
        make_chunk(doc, 1, "beta"),
        make_chunk(doc, 2, "epsilon"),
        make_chunk(doc, 3, "zeta"),
        make_chunk(doc, 4, "eta"),
        make_chunk(doc, 5, "gamma delta"),
    ]
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.4)]

    assert oracle_normalized_evidence_coverage_at_5(example, chunks, ranked) == 0.0


def test_normalized_coverage_returns_zero_without_evidence_chunks():
    doc = "alpha beta gamma delta epsilon"
    example = make_example(doc, "gamma")
    chunks = [make_chunk(doc, 0, "alpha beta"), make_chunk(doc, 1, "delta epsilon")]
    ranked = [(0, 0.9), (1, 0.8)]

    assert oracle_normalized_evidence_coverage_at_5(example, chunks, ranked) == 0.0


def test_build_fragmentation_pairs_averages_tied_low_efs_rows():
    results = [
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.0, "rr": 1.0},
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.0, "rr": 0.5},
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.4, "rr": 0.25},
    ]

    pairs = build_fragmentation_pairs(results)

    assert pairs == [
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "low_efs": 0.0,
            "high_efs": 0.4,
            "low_rr": 0.75,
            "high_rr": 0.25,
            "rr_difference": -0.5,
        }
    ]


def test_build_fragmentation_pairs_keeps_retrievers_separate():
    results = [
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.0, "rr": 1.0},
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.5, "rr": 0.5},
        {"query_id": "query-1", "retriever": "e5", "efs": 0.0, "rr": 0.5},
        {"query_id": "query-1", "retriever": "e5", "efs": 0.5, "rr": 1.0},
    ]

    pairs = build_fragmentation_pairs(results)

    assert len(pairs) == 2
    assert pairs[0]["retriever"] == "bm25"
    assert pairs[0]["rr_difference"] == -0.5
    assert pairs[1]["retriever"] == "e5"
    assert pairs[1]["rr_difference"] == 0.5


def test_build_fragmentation_pairs_accepts_csv_style_numbers():
    results = [
        {"query_id": "query-1", "retriever": "bm25", "efs": "0.0", "rr": "0.5"},
        {"query_id": "query-1", "retriever": "bm25", "efs": "0.25", "rr": "0.25"},
    ]

    pair = build_fragmentation_pairs(results)[0]

    assert pair["low_efs"] == 0.0
    assert pair["high_efs"] == 0.25
    assert pair["low_rr"] == 0.5
    assert pair["high_rr"] == 0.25
    assert pair["rr_difference"] == -0.25


def test_build_fragmentation_pairs_accepts_empty_results():
    assert build_fragmentation_pairs([]) == []


def test_build_fragmentation_pairs_skips_constant_efs_group():
    results = [
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.0, "rr": 1.0},
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.0, "rr": 0.5},
    ]

    assert build_fragmentation_pairs(results) == []


def test_bootstrap_mean_rr_difference_for_constant_effect():
    pairs = [
        {"query_id": f"query-{idx}", "retriever": "bm25", "rr_difference": -0.25}
        for idx in range(4)
    ]

    stat = bootstrap_mean_rr_difference(pairs, "bm25", n_resamples=100, seed=7)

    assert stat["n_queries"] == 4
    assert stat["mean_rr_difference"] == -0.25
    assert stat["ci_low"] == -0.25
    assert stat["ci_high"] == -0.25


def test_bootstrap_mean_rr_difference_keeps_retrievers_separate():
    pairs = [
        {"query_id": "query-1", "retriever": "bm25", "rr_difference": -0.5},
        {"query_id": "query-2", "retriever": "bm25", "rr_difference": 0.0},
        {"query_id": "query-1", "retriever": "e5", "rr_difference": 0.75},
    ]

    stat = bootstrap_mean_rr_difference(pairs, "bm25", n_resamples=100, seed=3)

    assert stat["n_queries"] == 2
    assert stat["mean_rr_difference"] == -0.25


def test_bootstrap_mean_rr_difference_is_reproducible():
    pairs = [
        {"query_id": "query-1", "retriever": "bm25", "rr_difference": -0.5},
        {"query_id": "query-2", "retriever": "bm25", "rr_difference": 0.0},
        {"query_id": "query-3", "retriever": "bm25", "rr_difference": 0.5},
    ]

    run1 = bootstrap_mean_rr_difference(pairs, "bm25", n_resamples=200, seed=11)
    run2 = bootstrap_mean_rr_difference(pairs, "bm25", n_resamples=200, seed=11)

    assert run1 == run2


def test_bootstrap_mean_rr_difference_rejects_missing_retriever():
    with pytest.raises(ValueError, match="no pairs found"):
        bootstrap_mean_rr_difference([], "bm25")


def test_bootstrap_mean_rr_difference_rejects_bad_settings():
    pairs = [{"query_id": "query-1", "retriever": "bm25", "rr_difference": -0.25}]

    with pytest.raises(ValueError, match="n_resamples must be positive"):
        bootstrap_mean_rr_difference(pairs, "bm25", n_resamples=0)

    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_mean_rr_difference(pairs, "bm25", confidence_level=1.0)


def _make_document_bootstrap_rows(
    query_id: str, document_id: str, retriever: str, low_rr: float, high_rr: float
) -> list[dict]:
    return [
        {
            "query_id": query_id,
            "document_id": document_id,
            "retriever": retriever,
            "efs": 0.0,
            "rr": low_rr,
        },
        {
            "query_id": query_id,
            "document_id": document_id,
            "retriever": retriever,
            "efs": 0.5,
            "rr": high_rr,
        },
    ]


def test_document_bootstrap_keeps_queries_from_same_document_together():
    results = []
    results.extend(_make_document_bootstrap_rows("query-1", "doc-a", "bm25", 1.0, 0.5))
    results.extend(_make_document_bootstrap_rows("query-2", "doc-a", "bm25", 1.0, 0.5))
    results.extend(_make_document_bootstrap_rows("query-3", "doc-b", "bm25", 0.5, 1.0))

    stat = bootstrap_mean_rr_difference_by_document(
        results, "bm25", n_resamples=1_000, seed=3
    )

    assert stat["n_queries"] == 3
    assert stat["n_documents"] == 2
    assert stat["mean_rr_difference"] == pytest.approx(-1 / 6)
    assert stat["ci_low"] == pytest.approx(-0.5)
    assert stat["ci_high"] == pytest.approx(0.5)


def test_document_bootstrap_observed_mean_matches_query_pairs():
    results = []
    results.extend(_make_document_bootstrap_rows("query-1", "doc-a", "bm25", 1.0, 0.75))
    results.extend(_make_document_bootstrap_rows("query-2", "doc-a", "bm25", 1.0, 0.5))
    results.extend(_make_document_bootstrap_rows("query-3", "doc-b", "bm25", 0.5, 1.0))

    pairs = build_fragmentation_pairs(results)
    expected = sum(row["rr_difference"] for row in pairs) / len(pairs)

    stat = bootstrap_mean_rr_difference_by_document(
        results, "bm25", n_resamples=100, seed=7
    )

    assert stat["mean_rr_difference"] == pytest.approx(expected)


def test_document_bootstrap_keeps_retrievers_separate():
    results = []
    results.extend(_make_document_bootstrap_rows("query-1", "doc-a", "bm25", 1.0, 0.5))
    results.extend(_make_document_bootstrap_rows("query-1", "doc-a", "e5", 0.5, 1.0))

    stat = bootstrap_mean_rr_difference_by_document(
        results, "bm25", n_resamples=100, seed=2
    )

    assert stat["n_queries"] == 1
    assert stat["n_documents"] == 1
    assert stat["mean_rr_difference"] == pytest.approx(-0.5)
    assert stat["ci_low"] == pytest.approx(-0.5)
    assert stat["ci_high"] == pytest.approx(-0.5)


def test_document_bootstrap_rejects_query_with_multiple_documents():
    results = _make_document_bootstrap_rows("query-1", "doc-a", "bm25", 1.0, 0.5)
    results[1]["document_id"] = "doc-b"

    with pytest.raises(ValueError, match="maps to multiple documents"):
        bootstrap_mean_rr_difference_by_document(results, "bm25", n_resamples=100)


def test_document_bootstrap_rejects_bad_settings_and_missing_retriever():
    results = _make_document_bootstrap_rows("query-1", "doc-a", "bm25", 1.0, 0.5)

    with pytest.raises(ValueError, match="n_resamples must be positive"):
        bootstrap_mean_rr_difference_by_document(results, "bm25", n_resamples=0)

    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_mean_rr_difference_by_document(results, "bm25", confidence_level=1.0)

    with pytest.raises(ValueError, match="no pairs found"):
        bootstrap_mean_rr_difference_by_document(results, "e5", n_resamples=100)


def test_within_query_spearman_finds_perfect_negative_association():
    results = [
        {"query_id": "query-1", "retriever": "bm25", "efs": efs, "rr": rr}
        for efs, rr in [(0.0, 1.0), (0.1, 0.75), (0.2, 0.5), (0.3, 0.25)]
    ]

    corr = build_within_query_spearman(results)[0]

    assert corr["n_phases"] == 4
    assert corr["spearman_rho"] == pytest.approx(-1.0)


def test_within_query_spearman_handles_ties():
    results = [
        {"query_id": "query-1", "retriever": "bm25", "efs": efs, "rr": rr}
        for efs, rr in [(0.0, 1.0), (0.0, 1.0), (0.5, 0.5), (0.5, 0.5)]
    ]

    corr = build_within_query_spearman(results)[0]

    assert corr["spearman_rho"] == pytest.approx(-1.0)


def test_within_query_spearman_is_undefined_for_constant_rr():
    results = [
        {"query_id": "query-1", "retriever": "bm25", "efs": efs, "rr": 1.0}
        for efs in [0.0, 0.1, 0.2]
    ]

    corr = build_within_query_spearman(results)[0]

    assert corr["spearman_rho"] is None


def test_within_query_spearman_keeps_retrievers_separate():
    results = [
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.0, "rr": 1.0},
        {"query_id": "query-1", "retriever": "bm25", "efs": 0.5, "rr": 0.5},
        {"query_id": "query-1", "retriever": "e5", "efs": 0.0, "rr": 0.5},
        {"query_id": "query-1", "retriever": "e5", "efs": 0.5, "rr": 1.0},
    ]

    corrs = build_within_query_spearman(results)

    assert len(corrs) == 2
    assert corrs[0]["retriever"] == "bm25"
    assert corrs[0]["spearman_rho"] == pytest.approx(-1.0)
    assert corrs[1]["retriever"] == "e5"
    assert corrs[1]["spearman_rho"] == pytest.approx(1.0)


def test_summarize_within_query_spearman_counts_undefined_queries():
    corrs = [
        {"query_id": "query-1", "retriever": "bm25", "spearman_rho": -1.0},
        {"query_id": "query-2", "retriever": "bm25", "spearman_rho": -0.5},
        {"query_id": "query-3", "retriever": "bm25", "spearman_rho": None},
    ]

    summary = summarize_within_query_spearman(corrs, "bm25")

    assert summary == {
        "retriever": "bm25",
        "n_queries": 3,
        "n_defined": 2,
        "n_undefined": 1,
        "mean_rho": -0.75,
        "median_rho": -0.75,
    }


def test_summarize_within_query_spearman_rejects_missing_retriever():
    with pytest.raises(ValueError, match="no correlations found"):
        summarize_within_query_spearman([], "bm25")


def test_build_efs_rr_plot_data_aggregates_identical_coordinates():
    results = [
        {"retriever": "bm25", "efs": 0.0, "rr": 1.0},
        {"retriever": "bm25", "efs": "0.0", "rr": "1.0"},
        {"retriever": "bm25", "efs": 0.25, "rr": 0.5},
    ]

    plot = build_efs_rr_plot_data(results, "bm25")

    assert plot == [
        {"efs": 0.0, "rr": 1.0, "count": 2},
        {"efs": 0.25, "rr": 0.5, "count": 1},
    ]


def test_build_efs_rr_plot_data_keeps_retrievers_separate():
    results = [
        {"retriever": "bm25", "efs": 0.0, "rr": 1.0},
        {"retriever": "e5", "efs": 0.0, "rr": 0.5},
    ]

    assert build_efs_rr_plot_data(results, "e5") == [
        {"efs": 0.0, "rr": 0.5, "count": 1}
    ]


def test_build_efs_rr_plot_data_rejects_missing_retriever():
    results = [{"retriever": "bm25", "efs": 0.0, "rr": 1.0}]

    with pytest.raises(ValueError, match="no results found for retriever: e5"):
        build_efs_rr_plot_data(results, "e5")


def test_full_index_fragmentation_pairs_average_tied_efs_rows():
    results = [
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "efs": "0.0",
            "ndcg_at_5": "1.0",
            "oracle_normalized_coverage_at_5": "1.0",
        },
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "efs": "0.0",
            "ndcg_at_5": "0.8",
            "oracle_normalized_coverage_at_5": "0.9",
        },
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "efs": "0.5",
            "ndcg_at_5": "0.5",
            "oracle_normalized_coverage_at_5": "0.7",
        },
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "efs": "0.5",
            "ndcg_at_5": "0.3",
            "oracle_normalized_coverage_at_5": "0.5",
        },
    ]

    pair = build_full_index_fragmentation_pairs(results)[0]

    assert pair["low_efs"] == 0.0
    assert pair["high_efs"] == 0.5
    assert pair["low_ndcg_at_5"] == pytest.approx(0.9)
    assert pair["high_ndcg_at_5"] == pytest.approx(0.4)
    assert pair["ndcg_at_5_difference"] == pytest.approx(-0.5)
    assert pair["low_oracle_normalized_coverage_at_5"] == pytest.approx(0.95)
    assert pair["high_oracle_normalized_coverage_at_5"] == pytest.approx(0.6)
    assert pair["oracle_normalized_coverage_at_5_difference"] == pytest.approx(-0.35)


def test_full_index_fragmentation_pairs_skip_constant_efs_group():
    results = [
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "efs": 0.0,
            "ndcg_at_5": 1.0,
            "oracle_normalized_coverage_at_5": 1.0,
        },
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "efs": 0.0,
            "ndcg_at_5": 0.6,
            "oracle_normalized_coverage_at_5": 0.8,
        },
    ]

    assert build_full_index_fragmentation_pairs(results) == []


def test_full_index_fragmentation_pairs_keep_retrievers_separate():
    results = [
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "efs": 0.0,
            "ndcg_at_5": 1.0,
            "oracle_normalized_coverage_at_5": 1.0,
        },
        {
            "query_id": "query-1",
            "retriever": "bm25",
            "efs": 0.5,
            "ndcg_at_5": 0.5,
            "oracle_normalized_coverage_at_5": 0.8,
        },
        {
            "query_id": "query-1",
            "retriever": "e5",
            "efs": 0.0,
            "ndcg_at_5": 0.7,
            "oracle_normalized_coverage_at_5": 0.9,
        },
        {
            "query_id": "query-1",
            "retriever": "e5",
            "efs": 0.5,
            "ndcg_at_5": 0.8,
            "oracle_normalized_coverage_at_5": 1.0,
        },
    ]

    pairs = build_full_index_fragmentation_pairs(results)

    assert len(pairs) == 2
    assert pairs[0]["retriever"] == "bm25"
    assert pairs[0]["ndcg_at_5_difference"] == pytest.approx(-0.5)
    assert pairs[1]["retriever"] == "e5"
    assert pairs[1]["ndcg_at_5_difference"] == pytest.approx(0.1)


def test_summarize_full_index_fragmentation_averages_query_pairs():
    pairs = [
        {
            "retriever": "bm25",
            "low_ndcg_at_5": 1.0,
            "high_ndcg_at_5": 0.5,
            "ndcg_at_5_difference": -0.5,
            "low_oracle_normalized_coverage_at_5": 1.0,
            "high_oracle_normalized_coverage_at_5": 0.8,
            "oracle_normalized_coverage_at_5_difference": -0.2,
        },
        {
            "retriever": "bm25",
            "low_ndcg_at_5": 0.8,
            "high_ndcg_at_5": 0.6,
            "ndcg_at_5_difference": -0.2,
            "low_oracle_normalized_coverage_at_5": 0.9,
            "high_oracle_normalized_coverage_at_5": 0.9,
            "oracle_normalized_coverage_at_5_difference": 0.0,
        },
    ]

    summary = summarize_full_index_fragmentation(pairs, "bm25")

    assert summary["n_queries"] == 2
    assert summary["mean_low_ndcg_at_5"] == pytest.approx(0.9)
    assert summary["mean_high_ndcg_at_5"] == pytest.approx(0.55)
    assert summary["mean_ndcg_at_5_difference"] == pytest.approx(-0.35)
    assert summary["mean_low_oracle_normalized_coverage_at_5"] == pytest.approx(0.95)
    assert summary["mean_high_oracle_normalized_coverage_at_5"] == pytest.approx(0.85)
    assert summary["mean_oracle_normalized_coverage_at_5_difference"] == pytest.approx(
        -0.1
    )


def test_summarize_full_index_fragmentation_rejects_missing_retriever():
    with pytest.raises(ValueError, match="no full-index pairs found"):
        summarize_full_index_fragmentation([], "bm25")


def _case_row(query_id, retriever, phase, efs, rr, doc_id="doc-1"):
    return {
        "query_id": query_id,
        "document_id": doc_id,
        "retriever": retriever,
        "phase": phase,
        "efs": efs,
        "rr": rr,
    }


def test_select_error_cases_rejects_nonpositive_case_count():
    with pytest.raises(ValueError, match="n_cases must be positive"):
        select_error_cases([], n_cases=0)


def test_select_error_cases_finds_strongest_failure():
    results = []

    for retriever, high_rr in (("bm25", 0.25), ("e5", 0.5)):
        results.extend(
            [
                _case_row("query-1", retriever, 0, 0.0, 1.0),
                _case_row("query-1", retriever, 25, 0.5, high_rr),
            ]
        )

    for retriever in ("bm25", "e5"):
        results.extend(
            [
                _case_row("query-2", retriever, 0, 0.0, 1.0),
                _case_row("query-2", retriever, 25, 0.4, 0.8),
            ]
        )

    fail = select_error_cases(results, n_cases=1)["failures"][0]

    assert fail["query_id"] == "query-1"
    assert fail["focus_retriever"] == "bm25"
    assert fail["bm25_rr_difference"] == pytest.approx(-0.75)
    assert fail["e5_rr_difference"] == pytest.approx(-0.5)


def test_select_error_cases_averages_tied_extreme_phases():
    results = []

    for retriever in ("bm25", "e5"):
        results.extend(
            [
                _case_row("query-1", retriever, 0, 0.0, 1.0),
                _case_row("query-1", retriever, 25, 0.0, 0.5),
                _case_row("query-1", retriever, 50, 0.5, 0.25),
            ]
        )

    fail = select_error_cases(results, n_cases=1)["failures"][0]

    assert fail["low_phases"] == (0, 25)
    assert fail["high_phases"] == (50,)
    assert fail["bm25_low_rr"] == pytest.approx(0.75)
    assert fail["bm25_rr_difference"] == pytest.approx(-0.5)


def test_select_error_cases_requires_both_retrievers_for_robust_case():
    results = [
        _case_row("query-1", "bm25", 0, 0.0, 1.0),
        _case_row("query-1", "bm25", 25, 0.5, 1.0),
        _case_row("query-1", "e5", 0, 0.0, 1.0),
        _case_row("query-1", "e5", 25, 0.5, 1.0),
        _case_row("query-2", "bm25", 0, 0.0, 1.0),
        _case_row("query-2", "bm25", 25, 0.5, 1.0),
    ]

    robust = select_error_cases(results, n_cases=3)["robust"]

    assert len(robust) == 1
    assert robust[0]["query_id"] == "query-1"
    assert robust[0]["bm25_high_rr"] == 1.0
    assert robust[0]["e5_high_rr"] == 1.0


def test_select_error_cases_prefers_largest_directional_disagreement():
    results = []

    values = {
        "query-1": {"bm25": (1.0, 0.25), "e5": (0.5, 1.0)},
        "query-2": {"bm25": (1.0, 0.5), "e5": (0.75, 1.0)},
    }

    for query_id, retrievers in values.items():
        for retriever, (low_rr, high_rr) in retrievers.items():
            results.extend(
                [
                    _case_row(query_id, retriever, 0, 0.0, low_rr),
                    _case_row(query_id, retriever, 25, 0.5, high_rr),
                ]
            )

    diff = select_error_cases(results, n_cases=1)["disagreements"][0]

    assert diff["query_id"] == "query-1"
    assert diff["bm25_rr_difference"] == pytest.approx(-0.75)
    assert diff["e5_rr_difference"] == pytest.approx(0.5)
    assert diff["rr_difference_gap"] == pytest.approx(1.25)


def _make_overlap_comparison_rows(
    query_id, retriever, efs_values, rr_values, ndcg_values, coverage_values
):
    return [
        {
            "query_id": query_id,
            "retriever": retriever,
            "phase": phase,
            "efs": efs,
            "rr": rr,
            "ndcg_at_5": ndcg,
            "oracle_normalized_coverage_at_5": coverage,
        }
        for phase, (efs, rr, ndcg, coverage) in enumerate(
            zip(efs_values, rr_values, ndcg_values, coverage_values)
        )
    ]


def test_overlap_fragmentation_comparison_deduplicates_retrievers():
    base = []
    overlap = []

    for retriever in ("bm25", "e5"):
        base.extend(
            _make_overlap_comparison_rows(
                "query-1", retriever, [0.0, 0.4], [1.0, 0.5], [1.0, 0.6], [1.0, 0.8]
            )
        )
        overlap.extend(
            _make_overlap_comparison_rows(
                "query-1", retriever, [0.0, 0.2], [1.0, 0.8], [1.0, 0.9], [1.0, 0.95]
            )
        )

    comparison = build_overlap_fragmentation_comparison(base, overlap)[0]

    assert comparison == {
        "query_id": "query-1",
        "n_phases": 2,
        "baseline_mean_efs": pytest.approx(0.2),
        "overlap_mean_efs": pytest.approx(0.1),
        "mean_efs_reduction": pytest.approx(0.1),
        "baseline_max_efs": pytest.approx(0.4),
        "overlap_max_efs": pytest.approx(0.2),
        "max_efs_reduction": pytest.approx(0.2),
    }


def test_overlap_fragmentation_comparison_rejects_mismatched_queries():
    base = [{"query_id": "query-1", "retriever": "bm25", "phase": 0, "efs": 0.0}]
    overlap = [{"query_id": "query-2", "retriever": "bm25", "phase": 0, "efs": 0.0}]

    with pytest.raises(ValueError, match="query sets do not match"):
        build_overlap_fragmentation_comparison(base, overlap)


def test_overlap_fragmentation_comparison_rejects_inconsistent_efs():
    base = [
        {"query_id": "query-1", "retriever": "bm25", "phase": 0, "efs": 0.1},
        {"query_id": "query-1", "retriever": "e5", "phase": 0, "efs": 0.2},
    ]

    with pytest.raises(ValueError, match="inconsistent EFS"):
        build_overlap_fragmentation_comparison(base, base)


def test_summarize_overlap_fragmentation_counts_query_changes():
    comparisons = [
        {
            "baseline_mean_efs": 0.30,
            "overlap_mean_efs": 0.20,
            "mean_efs_reduction": 0.10,
            "baseline_max_efs": 0.50,
            "overlap_max_efs": 0.30,
            "max_efs_reduction": 0.20,
        },
        {
            "baseline_mean_efs": 0.20,
            "overlap_mean_efs": 0.20,
            "mean_efs_reduction": 0.0,
            "baseline_max_efs": 0.40,
            "overlap_max_efs": 0.40,
            "max_efs_reduction": 0.0,
        },
        {
            "baseline_mean_efs": 0.10,
            "overlap_mean_efs": 0.15,
            "mean_efs_reduction": -0.05,
            "baseline_max_efs": 0.20,
            "overlap_max_efs": 0.25,
            "max_efs_reduction": -0.05,
        },
    ]

    summary = summarize_overlap_fragmentation(comparisons)

    assert summary["n_queries"] == 3
    assert summary["mean_efs_reduction"] == pytest.approx(0.05 / 3)
    assert summary["mean_max_efs_reduction"] == pytest.approx(0.15 / 3)
    assert summary["n_queries_reduced"] == 1
    assert summary["n_queries_unchanged"] == 1
    assert summary["n_queries_increased"] == 1


def test_bootstrap_mean_overlap_efs_reduction_for_constant_effect():
    comparisons = [{"mean_efs_reduction": 0.125} for _ in range(4)]

    stat = bootstrap_mean_overlap_efs_reduction(comparisons, n_resamples=100, seed=4)

    assert stat["n_queries"] == 4
    assert stat["mean_efs_reduction"] == pytest.approx(0.125)
    assert stat["ci_low"] == pytest.approx(0.125)
    assert stat["ci_high"] == pytest.approx(0.125)


def test_overlap_retrieval_comparison_measures_mitigation():
    base = []
    overlap = []

    for retriever in ("bm25", "e5"):
        base.extend(
            _make_overlap_comparison_rows(
                "query-1", retriever, [0.0, 0.4], [1.0, 0.5], [1.0, 0.6], [1.0, 0.8]
            )
        )
        overlap.extend(
            _make_overlap_comparison_rows(
                "query-1", retriever, [0.0, 0.2], [1.0, 0.8], [1.0, 0.9], [1.0, 0.95]
            )
        )

    comparisons = build_overlap_retrieval_comparison(base, overlap)

    assert len(comparisons) == 2

    for row in comparisons:
        assert row["baseline_rr_difference"] == pytest.approx(-0.5)
        assert row["overlap_rr_difference"] == pytest.approx(-0.2)
        assert row["rr_mitigation"] == pytest.approx(0.3)
        assert row["baseline_ndcg_at_5_difference"] == pytest.approx(-0.4)
        assert row["overlap_ndcg_at_5_difference"] == pytest.approx(-0.1)
        assert row["ndcg_at_5_mitigation"] == pytest.approx(0.3)
        assert row[
            "baseline_oracle_normalized_coverage_at_5_difference"
        ] == pytest.approx(-0.2)
        assert row[
            "overlap_oracle_normalized_coverage_at_5_difference"
        ] == pytest.approx(-0.05)
        assert row["oracle_normalized_coverage_at_5_mitigation"] == pytest.approx(0.15)


def test_overlap_retrieval_comparison_skips_undefined_overlap_contrast():
    base = []
    overlap = []

    for retriever in ("bm25", "e5"):
        base.extend(
            _make_overlap_comparison_rows(
                "query-1", retriever, [0.0, 0.4], [1.0, 0.5], [1.0, 0.6], [1.0, 0.8]
            )
        )
        overlap.extend(
            _make_overlap_comparison_rows(
                "query-1", retriever, [0.0, 0.0], [1.0, 0.5], [1.0, 0.6], [1.0, 0.8]
            )
        )

    assert build_overlap_retrieval_comparison(base, overlap) == []


def test_overlap_retrieval_comparison_uses_only_shared_defined_contrasts():
    base = []
    overlap = []

    for retriever in ("bm25", "e5"):
        base.extend(
            _make_overlap_comparison_rows(
                "query-1", retriever, [0.0, 0.4], [1.0, 0.5], [1.0, 0.6], [1.0, 0.8]
            )
        )
        overlap.extend(
            _make_overlap_comparison_rows(
                "query-1", retriever, [0.0, 0.2], [1.0, 0.8], [1.0, 0.9], [1.0, 0.95]
            )
        )

        base.extend(
            _make_overlap_comparison_rows(
                "query-2", retriever, [0.0, 0.5], [1.0, 0.25], [1.0, 0.5], [1.0, 0.7]
            )
        )
        overlap.extend(
            _make_overlap_comparison_rows(
                "query-2", retriever, [0.0, 0.0], [1.0, 0.5], [1.0, 0.8], [1.0, 0.9]
            )
        )

    comparisons = build_overlap_retrieval_comparison(base, overlap)

    assert len(comparisons) == 2
    assert {row["query_id"] for row in comparisons} == {"query-1"}
    assert {row["retriever"] for row in comparisons} == {"bm25", "e5"}


def test_overlap_retrieval_comparison_rejects_mismatched_raw_query_sets():
    base = _make_overlap_comparison_rows(
        "query-1", "bm25", [0.0, 0.4], [1.0, 0.5], [1.0, 0.6], [1.0, 0.8]
    )
    overlap = _make_overlap_comparison_rows(
        "query-2", "bm25", [0.0, 0.2], [1.0, 0.8], [1.0, 0.9], [1.0, 0.95]
    )

    with pytest.raises(ValueError, match="retrieval query sets do not match"):
        build_overlap_retrieval_comparison(base, overlap)


def test_summarize_overlap_retrieval_keeps_retrievers_separate():
    comparisons = [
        {
            "retriever": "bm25",
            "baseline_rr_difference": -0.4,
            "overlap_rr_difference": -0.1,
            "rr_mitigation": 0.3,
            "baseline_ndcg_at_5_difference": -0.2,
            "overlap_ndcg_at_5_difference": -0.1,
            "ndcg_at_5_mitigation": 0.1,
            "baseline_oracle_normalized_coverage_at_5_difference": -0.1,
            "overlap_oracle_normalized_coverage_at_5_difference": 0.0,
            "oracle_normalized_coverage_at_5_mitigation": 0.1,
        },
        {
            "retriever": "e5",
            "baseline_rr_difference": -0.2,
            "overlap_rr_difference": -0.3,
            "rr_mitigation": -0.1,
            "baseline_ndcg_at_5_difference": -0.1,
            "overlap_ndcg_at_5_difference": -0.2,
            "ndcg_at_5_mitigation": -0.1,
            "baseline_oracle_normalized_coverage_at_5_difference": 0.0,
            "overlap_oracle_normalized_coverage_at_5_difference": -0.1,
            "oracle_normalized_coverage_at_5_mitigation": -0.1,
        },
    ]

    summary = summarize_overlap_retrieval(comparisons, "bm25")

    assert summary["n_queries"] == 1
    assert summary["mean_rr_mitigation"] == pytest.approx(0.3)
    assert summary["mean_ndcg_at_5_mitigation"] == pytest.approx(0.1)
    assert summary["mean_oracle_normalized_coverage_at_5_mitigation"] == pytest.approx(
        0.1
    )


def test_bootstrap_mean_overlap_rr_mitigation_for_constant_effect():
    comparisons = [{"retriever": "bm25", "rr_mitigation": 0.2} for _ in range(5)]

    stat = bootstrap_mean_overlap_rr_mitigation(
        comparisons, "bm25", n_resamples=100, seed=5
    )

    assert stat["n_queries"] == 5
    assert stat["mean_rr_mitigation"] == pytest.approx(0.2)
    assert stat["ci_low"] == pytest.approx(0.2)
    assert stat["ci_high"] == pytest.approx(0.2)


def test_overlap_analysis_rejects_missing_data():
    with pytest.raises(ValueError, match="no overlap fragmentation"):
        summarize_overlap_fragmentation([])

    with pytest.raises(ValueError, match="no overlap fragmentation"):
        bootstrap_mean_overlap_efs_reduction([])

    with pytest.raises(ValueError, match="no overlap retrieval"):
        summarize_overlap_retrieval([], "bm25")

    with pytest.raises(ValueError, match="no overlap retrieval"):
        bootstrap_mean_overlap_rr_mitigation([], "bm25")
