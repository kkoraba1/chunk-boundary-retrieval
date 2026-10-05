import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from document_chunking_utility.chunkers import TextChunk

from boundary_retrieval.data import build_examples, load_dialogues, load_docs
from boundary_retrieval.evaluation import (
    best_evidence_chunk_index,
    build_efs_rr_plot_data,
    evidence_fragmentation_score,
    evidence_overlap_fraction,
    graded_evidence_relevance,
    graded_ndcg_at_5,
    oracle_normalized_evidence_coverage_at_5,
    reciprocal_rank,
    select_error_cases,
)
from boundary_retrieval.retrieval import load_e5, rank_bm25, rank_e5
from boundary_retrieval.segmentation import chunk_shifted

CHUNK_SIZE = 200
OVERLAP = 0
PHASES = (0, 25, 50, 75, 100, 125, 150, 175)
OVERLAP_ABLATION_OVERLAP = 40
OVERLAP_ABLATION_PHASES = (0, 40, 80, 120)
TOP_K = 5
DEV_PILOT_SIZE = 5

RESULT_FIELDS = (
    "query_id",
    "document_id",
    "chunk_size",
    "overlap",
    "phase",
    "retriever",
    "top_k",
    "efs",
    "rr",
    "ndcg_at_5",
    "oracle_normalized_coverage_at_5",
    "best_chunk_index",
    "distractor_count",
)


def build_phase_chunks(example: dict[str, Any]) -> dict[int, list[TextChunk]]:
    """Build chunks for each primary phase of one example."""
    return {
        phase: chunk_shifted(
            example["document_text"],
            chunk_size=CHUNK_SIZE,
            phase=phase,
            overlap=OVERLAP,
        )
        for phase in PHASES
    }


def build_overlap_ablation_chunks(
    example: dict[str, Any],
) -> dict[int, list[TextChunk]]:
    """Build chunks for the overlap ablation's four phases."""
    return {
        phase: chunk_shifted(
            example["document_text"],
            chunk_size=CHUNK_SIZE,
            phase=phase,
            overlap=OVERLAP_ABLATION_OVERLAP,
        )
        for phase in OVERLAP_ABLATION_PHASES
    }


def build_phase_evidence_info(
    example: dict[str, Any], phase_chunks: dict[int, list[TextChunk]]
) -> dict[int, dict[str, Any]]:
    """Find fragmentation and the best evidence chunk for each phase."""
    phase_info = {}

    for phase, chunks in phase_chunks.items():
        best_idx = best_evidence_chunk_index(example, chunks)

        phase_info[phase] = {
            "efs": evidence_fragmentation_score(example, chunks),
            "best_chunk_index": best_idx,
            "best_chunk": chunks[best_idx],
        }

    return phase_info


def build_fixed_distractors(
    example: dict[str, Any], phase_chunks: dict[int, list[TextChunk]]
) -> list[TextChunk]:
    """Build the fixed non-evidence distractor set from phase 0."""
    phase0_chunks = phase_chunks[PHASES[0]]

    distractors = [
        chunk
        for chunk in phase0_chunks
        if evidence_overlap_fraction(example, chunk) == 0.0
    ]

    if not distractors:
        raise ValueError("example has no non-evidence distractor chunks")

    return distractors


def build_canonical_distractors(example: dict[str, Any]) -> list[TextChunk]:
    """Build zero-overlap phase-0 distractors for the overlap ablation."""
    phase0_chunks = {
        0: chunk_shifted(
            example["document_text"], chunk_size=CHUNK_SIZE, phase=0, overlap=OVERLAP
        )
    }

    return build_fixed_distractors(example, phase0_chunks)


def run_bm25_controlled_probe(
    example: dict[str, Any],
    phase_info: dict[int, dict[str, Any]],
    distractors: list[TextChunk],
) -> dict[int, dict[str, Any]]:
    """Rank each phase's evidence chunk against the fixed distractors."""
    results = {}

    for phase, info in phase_info.items():
        candidates = [info["best_chunk"], *distractors]
        passages = [chunk.chunk_text for chunk in candidates]

        ranking = rank_bm25(example["query"], passages)

        results[phase] = {
            "ranking": ranking,
            "rr": reciprocal_rank(ranking, target_index=0),
        }

    return results


def run_e5_controlled_probe(
    example: dict[str, Any],
    phase_info: dict[int, dict[str, Any]],
    distractors: list[TextChunk],
    model: Any,
) -> dict[int, dict[str, Any]]:
    """Rank each phase's evidence chunk with the loaded E5 model."""
    results = {}

    for phase, info in phase_info.items():
        candidates = [info["best_chunk"], *distractors]
        passages = [chunk.chunk_text for chunk in candidates]

        ranking = rank_e5(example["query"], passages, model)

        results[phase] = {
            "ranking": ranking,
            "rr": reciprocal_rank(ranking, target_index=0),
        }

    return results


def run_bm25_full_index_validation(
    example: dict[str, Any], phase_chunks: dict[int, list[TextChunk]]
) -> dict[int, dict[str, Any]]:
    """Rank every shifted chunk with BM25 and calculate validation metrics."""
    results = {}

    for phase, chunks in phase_chunks.items():
        passages = [chunk.chunk_text for chunk in chunks]
        ranking = rank_bm25(example["query"], passages)
        rel = graded_evidence_relevance(example, chunks)

        results[phase] = {
            "ranking": ranking,
            "ndcg_at_5": graded_ndcg_at_5(ranking, rel),
            "oracle_normalized_coverage_at_5": (
                oracle_normalized_evidence_coverage_at_5(example, chunks, ranking)
            ),
        }

    return results


def run_e5_full_index_validation(
    example: dict[str, Any], phase_chunks: dict[int, list[TextChunk]], model: Any
) -> dict[int, dict[str, Any]]:
    """Rank every shifted chunk with E5 and calculate validation metrics."""
    results = {}

    for phase, chunks in phase_chunks.items():
        passages = [chunk.chunk_text for chunk in chunks]
        ranking = rank_e5(example["query"], passages, model)
        rel = graded_evidence_relevance(example, chunks)

        results[phase] = {
            "ranking": ranking,
            "ndcg_at_5": graded_ndcg_at_5(ranking, rel),
            "oracle_normalized_coverage_at_5": (
                oracle_normalized_evidence_coverage_at_5(example, chunks, ranking)
            ),
        }

    return results


def _run_example_phases(
    example: dict[str, Any],
    phase_chunks: dict[int, list[TextChunk]],
    distractors: list[TextChunk],
    overlap: int,
    model: Any,
) -> list[dict[str, Any]]:
    """Run both retrievers and collect result rows for the supplied phases."""
    phase_info = build_phase_evidence_info(example, phase_chunks)

    bm25_probe = run_bm25_controlled_probe(example, phase_info, distractors)
    e5_probe = run_e5_controlled_probe(example, phase_info, distractors, model)

    bm25_full = run_bm25_full_index_validation(example, phase_chunks)
    e5_full = run_e5_full_index_validation(example, phase_chunks, model)

    results = []

    for phase in phase_chunks:
        for retriever, probe, full in (
            ("bm25", bm25_probe, bm25_full),
            ("e5", e5_probe, e5_full),
        ):
            results.append(
                {
                    "query_id": example["query_id"],
                    "document_id": example["document_id"],
                    "chunk_size": CHUNK_SIZE,
                    "overlap": overlap,
                    "phase": phase,
                    "retriever": retriever,
                    "top_k": TOP_K,
                    "efs": phase_info[phase]["efs"],
                    "rr": probe[phase]["rr"],
                    "ndcg_at_5": full[phase]["ndcg_at_5"],
                    "oracle_normalized_coverage_at_5": full[phase][
                        "oracle_normalized_coverage_at_5"
                    ],
                    "best_chunk_index": phase_info[phase]["best_chunk_index"],
                    "distractor_count": len(distractors),
                }
            )

    return results


def run_example(example: dict[str, Any], model: Any) -> list[dict[str, Any]]:
    """Run both retrieval experiments for one example."""
    phase_chunks = build_phase_chunks(example)
    distractors = build_fixed_distractors(example, phase_chunks)

    return _run_example_phases(example, phase_chunks, distractors, OVERLAP, model)


def run_overlap_ablation_example(
    example: dict[str, Any], model: Any
) -> list[dict[str, Any]]:
    """Run the 40-word overlap ablation for one example."""
    phase_chunks = build_overlap_ablation_chunks(example)
    distractors = build_canonical_distractors(example)

    return _run_example_phases(
        example, phase_chunks, distractors, OVERLAP_ABLATION_OVERLAP, model
    )


def run_overlap_ablation(
    val_examples: list[dict[str, Any]], model: Any
) -> list[dict[str, Any]]:
    """Run the overlap ablation on all held-out validation examples."""
    results = []

    for example in val_examples:
        results.extend(run_overlap_ablation_example(example, model))

    return results


def run_dev_pilot(
    train_examples: list[dict[str, Any]], model: Any
) -> list[dict[str, Any]]:
    """Run the experiment on examples from five distinct training documents."""
    selected_examples = []
    seen_doc_ids = set()

    for example in train_examples:
        doc_id = example["document_id"]

        if doc_id in seen_doc_ids:
            continue

        selected_examples.append(example)
        seen_doc_ids.add(doc_id)

        if len(selected_examples) == DEV_PILOT_SIZE:
            break

    if len(selected_examples) < DEV_PILOT_SIZE:
        raise ValueError(
            "not enough distinct training documents for the development pilot"
        )

    results = []

    for example in selected_examples:
        results.extend(run_example(example, model))

    return results


def run_primary_experiment(
    val_examples: list[dict[str, Any]], model: Any
) -> list[dict[str, Any]]:
    """Run the primary experiment on all held-out validation examples."""
    results = []

    for example in val_examples:
        results.extend(run_example(example, model))

    return results


def _validate_results(
    results: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    phases: tuple[int, ...],
    overlap: int,
    result_name: str,
) -> None:
    """Check result fields and require every expected query/phase/retriever row."""
    expected_docs = {}

    for example in val_examples:
        query_id = example["query_id"]

        if query_id in expected_docs:
            raise ValueError(f"duplicate validation query_id: {query_id}")

        expected_docs[query_id] = example["document_id"]

    expected_conditions = {
        (phase, retriever) for phase in phases for retriever in ("bm25", "e5")
    }

    expected_row_count = len(val_examples) * len(expected_conditions)

    if len(results) != expected_row_count:
        raise ValueError(
            f"unexpected {result_name} result count: "
            f"expected {expected_row_count}, got {len(results)}"
        )

    seen_conditions = set()
    conditions_by_query = {query_id: set() for query_id in expected_docs}
    phase_meta = {}
    distractor_counts = {}

    for row in results:
        missing_fields = [field for field in RESULT_FIELDS if field not in row]

        if missing_fields:
            raise ValueError(
                f"{result_name} result row is missing fields: "
                + ", ".join(missing_fields)
            )

        query_id = row["query_id"]

        if query_id not in expected_docs:
            raise ValueError(
                f"unexpected query_id in {result_name} results: {query_id}"
            )

        if row["document_id"] != expected_docs[query_id]:
            raise ValueError(f"document_id does not match query {query_id}")

        if row["chunk_size"] != CHUNK_SIZE:
            raise ValueError(f"unexpected chunk_size for query {query_id}")

        if row["overlap"] != overlap:
            raise ValueError(f"unexpected overlap for query {query_id}")

        if row["top_k"] != TOP_K:
            raise ValueError(f"unexpected top_k for query {query_id}")

        phase = row["phase"]
        retriever = row["retriever"]

        if phase not in phases:
            raise ValueError(f"unexpected phase for query {query_id}: {phase}")

        if retriever not in {"bm25", "e5"}:
            raise ValueError(f"unexpected retriever for query {query_id}: {retriever}")

        for field in ("efs", "rr", "ndcg_at_5", "oracle_normalized_coverage_at_5"):
            value = row[field]

            if not isinstance(value, (int, float)):
                raise TypeError(f"{field} is not numeric for query {query_id}")

            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} is outside [0, 1] for query {query_id}")

        if not isinstance(row["best_chunk_index"], int) or row["best_chunk_index"] < 0:
            raise ValueError(f"invalid best_chunk_index for query {query_id}")

        if not isinstance(row["distractor_count"], int) or row["distractor_count"] <= 0:
            raise ValueError(f"invalid distractor_count for query {query_id}")

        phase_key = (query_id, phase)

        if phase_key in phase_meta:
            expected_efs, expected_best_chunk_idx = phase_meta[phase_key]

            if row["efs"] != expected_efs:
                raise ValueError(
                    f"EFS differs between retrievers for query {query_id}, phase {phase}"
                )

            if row["best_chunk_index"] != expected_best_chunk_idx:
                raise ValueError(
                    "best_chunk_index differs between retrievers for "
                    f"query {query_id}, phase {phase}"
                )
        else:
            phase_meta[phase_key] = (row["efs"], row["best_chunk_index"])

        if query_id in distractor_counts:
            if row["distractor_count"] != distractor_counts[query_id]:
                raise ValueError(
                    f"distractor_count changes across conditions for query {query_id}"
                )
        else:
            distractor_counts[query_id] = row["distractor_count"]

        condition = (query_id, phase, retriever)

        if condition in seen_conditions:
            raise ValueError(
                f"duplicate {result_name} result condition: "
                f"{query_id}, phase {phase}, {retriever}"
            )

        seen_conditions.add(condition)
        conditions_by_query[query_id].add((phase, retriever))

    for query_id, conditions in conditions_by_query.items():
        if conditions != expected_conditions:
            raise ValueError(f"incomplete conditions for query {query_id}")


def validate_primary_results(
    results: list[dict[str, Any]], val_examples: list[dict[str, Any]]
) -> None:
    """Check that the held-out primary result table is valid."""
    _validate_results(results, val_examples, PHASES, OVERLAP, "primary")


def validate_overlap_ablation_results(
    results: list[dict[str, Any]], val_examples: list[dict[str, Any]]
) -> None:
    """Check that the overlap ablation result table is valid."""
    _validate_results(
        results,
        val_examples,
        OVERLAP_ABLATION_PHASES,
        OVERLAP_ABLATION_OVERLAP,
        "overlap ablation",
    )


def _save_results(
    results: list[dict[str, Any]], output_path: str | Path, result_name: str
) -> Path:
    """Save validated rows using the experiment's CSV columns."""
    if not results:
        raise ValueError(f"cannot save empty {result_name} results")

    for row in results:
        missing_fields = [field for field in RESULT_FIELDS if field not in row]

        if missing_fields:
            raise ValueError(
                f"{result_name} result row is missing fields: "
                + ", ".join(missing_fields)
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    return output_path


def save_primary_results(
    results: list[dict[str, Any]], output_path: str | Path
) -> Path:
    """Save the primary result rows as a CSV file."""
    return _save_results(results, output_path, "primary")


def save_overlap_ablation_results(
    results: list[dict[str, Any]], output_path: str | Path
) -> Path:
    """Save the overlap ablation result rows as a CSV file."""
    return _save_results(results, output_path, "overlap ablation")


def load_training_examples(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load the filtered Doc2Dial training examples."""
    data_dir = Path(data_dir)

    docs = load_docs(data_dir / "doc2dial_doc.json")
    train_dialogues = load_dialogues(data_dir / "doc2dial_dial_train.json")

    return build_examples(train_dialogues, docs, chunk_size=CHUNK_SIZE, phases=PHASES)


def load_validation_examples(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load the filtered Doc2Dial validation examples."""
    data_dir = Path(data_dir)

    docs = load_docs(data_dir / "doc2dial_doc.json")
    val_dialogues = load_dialogues(data_dir / "doc2dial_dial_validation.json")

    return build_examples(val_dialogues, docs, chunk_size=CHUNK_SIZE, phases=PHASES)


def run_dev_pilot_from_data_dir(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load the training data and E5 model, then run the pilot."""
    train_examples = load_training_examples(data_dir)
    model = load_e5()

    return run_dev_pilot(train_examples, model)


def run_overlap_ablation_from_data_dir(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load validation data and E5, then run the overlap ablation."""
    val_examples = load_validation_examples(data_dir)
    model = load_e5()

    return run_overlap_ablation(val_examples, model)


def run_primary_experiment_from_data_dir(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load the validation data and E5 model, then run the primary experiment."""
    val_examples = load_validation_examples(data_dir)
    model = load_e5()

    return run_primary_experiment(val_examples, model)


def run_and_save_primary_experiment(
    data_dir: str | Path, output_path: str | Path
) -> Path:
    """Run, validate, and save the held-out primary experiment."""
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(f"primary results already exist: {output_path}")

    val_examples = load_validation_examples(data_dir)
    model = load_e5()
    results = run_primary_experiment(val_examples, model)

    validate_primary_results(results, val_examples)

    return save_primary_results(results, output_path)


def run_and_save_overlap_ablation(
    data_dir: str | Path, output_path: str | Path
) -> Path:
    """Run, validate, and save the held-out overlap ablation."""
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(f"overlap ablation results already exist: {output_path}")

    val_examples = load_validation_examples(data_dir)
    model = load_e5()
    results = run_overlap_ablation(val_examples, model)

    validate_overlap_ablation_results(results, val_examples)

    return save_overlap_ablation_results(results, output_path)


def gen_efs_rr_figure(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Generate the main controlled-probe EFS-versus-RR figure."""
    plot_data = {
        retriever: build_efs_rr_plot_data(results, retriever)
        for retriever in ("bm25", "e5")
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

    for axis, retriever in zip(axes, ("bm25", "e5")):
        rows = plot_data[retriever]
        sizes = [20 + 12 * math.sqrt(row["count"]) for row in rows]

        axis.scatter(
            [row["efs"] for row in rows],
            [row["rr"] for row in rows],
            s=sizes,
            alpha=0.65,
        )
        axis.set_title(retriever.upper())
        axis.set_xlabel("Evidence Fragmentation Score")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Reciprocal Rank")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def build_error_case_records(
    results: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    n_cases: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Rebuild selected cases with their queries, evidence, passages, and scores."""
    selected_cases = select_error_cases(results, n_cases=n_cases)

    examples_by_query = {}

    for example in val_examples:
        query_id = example["query_id"]

        if query_id in examples_by_query:
            raise ValueError(f"duplicate validation query_id: {query_id}")

        examples_by_query[query_id] = example

    results_lookup = {
        (row["query_id"], row["retriever"], int(row["phase"])): row for row in results
    }

    records = {"failures": [], "robust": [], "disagreements": []}

    for category, cases in selected_cases.items():
        for case in cases:
            query_id = case["query_id"]

            if query_id not in examples_by_query:
                raise ValueError(f"validation example not found for query: {query_id}")

            example = examples_by_query[query_id]

            if example["document_id"] != case["document_id"]:
                raise ValueError(f"document_id does not match query {query_id}")

            phase_chunks = build_phase_chunks(example)
            phase_info = build_phase_evidence_info(example, phase_chunks)

            conditions_by_level = {}

            for level in ("low", "high"):
                conditions = []

                for phase in case[f"{level}_phases"]:
                    bm25_key = (query_id, "bm25", phase)
                    e5_key = (query_id, "e5", phase)

                    if bm25_key not in results_lookup or e5_key not in results_lookup:
                        raise ValueError(
                            f"missing result row for query {query_id}, phase {phase}"
                        )

                    bm25_row = results_lookup[bm25_key]
                    e5_row = results_lookup[e5_key]
                    info = phase_info[phase]

                    for row in (bm25_row, e5_row):
                        if not math.isclose(float(row["efs"]), info["efs"]):
                            raise ValueError(
                                "stored EFS does not match reconstructed "
                                f"phase for query {query_id}"
                            )

                        if int(row["best_chunk_index"]) != info["best_chunk_index"]:
                            raise ValueError(
                                "stored best chunk does not match reconstructed "
                                f"phase for query {query_id}"
                            )

                    best_chunk = info["best_chunk"]

                    conditions.append(
                        {
                            "phase": phase,
                            "efs": info["efs"],
                            "best_chunk_index": info["best_chunk_index"],
                            "evidence_overlap_fraction": (
                                evidence_overlap_fraction(example, best_chunk)
                            ),
                            "passage": best_chunk.chunk_text,
                            "bm25_rr": float(bm25_row["rr"]),
                            "e5_rr": float(e5_row["rr"]),
                        }
                    )

                conditions_by_level[level] = conditions

            record = {
                "category": category,
                "query_id": query_id,
                "document_id": case["document_id"],
                "query": example["query"],
                "evidence_text": example["document_text"][
                    example["evidence_start"] : example["evidence_end"]
                ],
                "low_efs": case["low_efs"],
                "high_efs": case["high_efs"],
                "bm25_rr_difference": case["bm25_rr_difference"],
                "e5_rr_difference": case["e5_rr_difference"],
                "low_conditions": conditions_by_level["low"],
                "high_conditions": conditions_by_level["high"],
            }

            if "focus_retriever" in case:
                record["focus_retriever"] = case["focus_retriever"]

            if "rr_difference_gap" in case:
                record["rr_difference_gap"] = case["rr_difference_gap"]

            records[category].append(record)

    return records


def save_error_case_records(
    records: dict[str, list[dict[str, Any]]], output_path: str | Path
) -> Path:
    """Save the rebuilt error cases as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return output_path


def main(argv: list[str] | None = None) -> int:
    """Run a held-out experiment and save its results from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the chunk-boundary retrieval experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    primary_parser = subparsers.add_parser(
        "primary", help="Run the primary experiment."
    )
    primary_parser.add_argument(
        "data_dir", type=Path, help="Directory containing the Doc2Dial JSON files."
    )
    primary_parser.add_argument(
        "output_path", type=Path, help="CSV path for the primary results."
    )

    overlap_parser = subparsers.add_parser(
        "overlap", help="Run and save the held-out 40-word overlap ablation."
    )
    overlap_parser.add_argument(
        "data_dir", type=Path, help="Directory containing the Doc2Dial JSON files."
    )
    overlap_parser.add_argument(
        "output_path", type=Path, help="CSV path for the overlap-ablation results."
    )

    args = parser.parse_args(argv)

    if args.command == "primary":
        saved = run_and_save_primary_experiment(args.data_dir, args.output_path)
    else:
        saved = run_and_save_overlap_ablation(args.data_dir, args.output_path)

    print(f"Saved results: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
