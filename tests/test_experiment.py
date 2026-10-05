import csv
import json
from pathlib import Path

import pytest

from boundary_retrieval import experiment
from boundary_retrieval.experiment import (
    CHUNK_SIZE,
    DEV_PILOT_SIZE,
    OVERLAP,
    OVERLAP_ABLATION_OVERLAP,
    OVERLAP_ABLATION_PHASES,
    PHASES,
    RESULT_FIELDS,
    TOP_K,
    build_canonical_distractors,
    build_error_case_records,
    build_fixed_distractors,
    build_overlap_ablation_chunks,
    build_phase_chunks,
    build_phase_evidence_info,
    gen_efs_rr_figure,
    load_training_examples,
    load_validation_examples,
    run_and_save_overlap_ablation,
    run_and_save_primary_experiment,
    run_bm25_controlled_probe,
    run_bm25_full_index_validation,
    run_dev_pilot,
    run_dev_pilot_from_data_dir,
    run_e5_controlled_probe,
    run_e5_full_index_validation,
    run_example,
    run_overlap_ablation,
    run_overlap_ablation_example,
    run_overlap_ablation_from_data_dir,
    run_primary_experiment,
    run_primary_experiment_from_data_dir,
    save_error_case_records,
    save_overlap_ablation_results,
    save_primary_results,
    validate_overlap_ablation_results,
    validate_primary_results,
)


def test_primary_experiment_settings():
    assert CHUNK_SIZE == 200
    assert OVERLAP == 0
    assert PHASES == (0, 25, 50, 75, 100, 125, 150, 175)
    assert OVERLAP_ABLATION_OVERLAP == 40
    assert OVERLAP_ABLATION_PHASES == (0, 40, 80, 120)
    assert TOP_K == 5
    assert DEV_PILOT_SIZE == 5


def test_build_phase_chunks_builds_every_configured_phase():
    doc = " ".join(f"word{i}" for i in range(500))
    example = {"document_text": doc}

    phase_chunks = build_phase_chunks(example)

    assert tuple(phase_chunks) == PHASES
    assert phase_chunks[0][0].chunk_text == " ".join(
        f"word{i}" for i in range(CHUNK_SIZE)
    )
    assert phase_chunks[25][0].chunk_text == " ".join(f"word{i}" for i in range(25))
    assert all(chunks for chunks in phase_chunks.values())


def test_build_overlap_ablation_chunks_uses_reduced_overlap_cycle():
    words = [f"word{i}" for i in range(500)]
    doc = " ".join(words)
    example = {"document_text": doc}

    phase_chunks = build_overlap_ablation_chunks(example)

    assert tuple(phase_chunks) == OVERLAP_ABLATION_PHASES
    assert phase_chunks[0][0].chunk_text == " ".join(words[:CHUNK_SIZE])
    assert phase_chunks[40][0].chunk_text == " ".join(words[:80])
    assert phase_chunks[40][1].chunk_text == " ".join(words[40:240])
    assert phase_chunks[80][0].chunk_text == " ".join(words[:120])
    assert phase_chunks[120][0].chunk_text == " ".join(words[:160])


def test_build_phase_evidence_info_tracks_fragmentation_by_phase():
    doc = " ".join(f"word{i}" for i in range(500))
    evidence = " ".join(f"word{i}" for i in range(195, 205))
    evidence_start = doc.index(evidence)

    example = {
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    phase_chunks = build_phase_chunks(example)
    phase_info = build_phase_evidence_info(example, phase_chunks)

    assert tuple(phase_info) == PHASES
    assert phase_info[0]["efs"] == pytest.approx(0.5)
    assert phase_info[25]["efs"] == 0.0
    assert phase_info[0]["best_chunk_index"] == 0
    assert evidence.split()[0] in phase_info[0]["best_chunk"].chunk_text
    assert evidence in phase_info[25]["best_chunk"].chunk_text


def test_build_fixed_distractors_uses_phase_zero_non_evidence_chunks():
    doc = " ".join(f"word{i}" for i in range(600))
    evidence = " ".join(f"word{i}" for i in range(250, 260))
    evidence_start = doc.index(evidence)

    example = {
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    phase_chunks = build_phase_chunks(example)
    distractors = build_fixed_distractors(example, phase_chunks)

    assert len(distractors) == 2
    assert distractors[0] is phase_chunks[0][0]
    assert distractors[1] is phase_chunks[0][2]
    assert evidence not in distractors[0].chunk_text
    assert evidence not in distractors[1].chunk_text


def test_build_canonical_distractors_matches_primary_phase_zero():
    doc = " ".join(f"word{i}" for i in range(600))
    evidence = " ".join(f"word{i}" for i in range(250, 260))
    evidence_start = doc.index(evidence)

    example = {
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    primary_chunks = build_phase_chunks(example)
    expected = build_fixed_distractors(example, primary_chunks)
    got = build_canonical_distractors(example)

    assert got == expected
    assert [chunk.chunk_index for chunk in got] == [0, 2]


def test_build_fixed_distractors_rejects_example_without_distractors():
    doc = " ".join(f"word{i}" for i in range(100))
    evidence = " ".join(f"word{i}" for i in range(40, 50))
    evidence_start = doc.index(evidence)

    example = {
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    phase_chunks = build_phase_chunks(example)

    with pytest.raises(
        ValueError, match="example has no non-evidence distractor chunks"
    ):
        build_fixed_distractors(example, phase_chunks)


def test_run_bm25_controlled_probe_ranks_evidence_for_every_phase():
    words = [f"word{i}" for i in range(600)]
    words[250] = "needle"
    doc = " ".join(words)

    evidence = " ".join(words[250:260])
    evidence_start = doc.index(evidence)

    example = {
        "query": "needle",
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    phase_chunks = build_phase_chunks(example)
    phase_info = build_phase_evidence_info(example, phase_chunks)
    distractors = build_fixed_distractors(example, phase_chunks)

    results = run_bm25_controlled_probe(example, phase_info, distractors)

    assert tuple(results) == PHASES

    for phase in PHASES:
        assert results[phase]["ranking"][0][0] == 0
        assert results[phase]["rr"] == 1.0


def test_run_e5_controlled_probe_reuses_fixed_candidates_and_model(monkeypatch):
    doc = " ".join(f"word{i}" for i in range(600))
    evidence = " ".join(f"word{i}" for i in range(250, 260))
    evidence_start = doc.index(evidence)

    example = {
        "query": "test query",
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    phase_chunks = build_phase_chunks(example)
    phase_info = build_phase_evidence_info(example, phase_chunks)
    distractors = build_fixed_distractors(example, phase_chunks)

    model = object()
    calls = []

    def fake_rank_e5(query, passages, recv_model):
        calls.append({"query": query, "passages": list(passages), "model": recv_model})

        return [(idx, float(len(passages) - idx)) for idx in range(len(passages))]

    monkeypatch.setattr(experiment, "rank_e5", fake_rank_e5)

    results = run_e5_controlled_probe(example, phase_info, distractors, model)

    assert tuple(results) == PHASES
    assert len(calls) == len(PHASES)

    distractor_texts = [chunk.chunk_text for chunk in distractors]

    for phase, call in zip(PHASES, calls):
        assert call["query"] == example["query"]
        assert call["model"] is model
        assert call["passages"][0] == phase_info[phase]["best_chunk"].chunk_text
        assert call["passages"][1:] == distractor_texts
        assert results[phase]["ranking"][0][0] == 0
        assert results[phase]["rr"] == 1.0


def test_run_bm25_full_index_validation_scores_every_phase():
    words = [f"word{i}" for i in range(600)]
    words[250] = "needle"
    doc = " ".join(words)

    evidence = " ".join(words[250:260])
    evidence_start = doc.index(evidence)

    example = {
        "query": "needle",
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    phase_chunks = build_phase_chunks(example)

    results = run_bm25_full_index_validation(example, phase_chunks)

    assert tuple(results) == PHASES

    for phase in PHASES:
        ranking = results[phase]["ranking"]
        best_idx = next(
            idx
            for idx, chunk in enumerate(phase_chunks[phase])
            if "needle" in chunk.chunk_text
        )

        assert ranking[0][0] == best_idx
        assert results[phase]["ndcg_at_5"] == pytest.approx(1.0)
        assert results[phase]["oracle_normalized_coverage_at_5"] == pytest.approx(1.0)


def test_run_e5_full_index_validation_scores_every_phase(monkeypatch):
    words = [f"word{i}" for i in range(600)]
    words[250] = "needle"
    doc = " ".join(words)

    evidence = " ".join(words[250:260])
    evidence_start = doc.index(evidence)

    example = {
        "query": "needle",
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    phase_chunks = build_phase_chunks(example)
    model = object()
    calls = []

    def fake_rank_e5(query, passages, recv_model):
        calls.append({"query": query, "passages": list(passages), "model": recv_model})

        evidence_idx = next(
            idx for idx, passage in enumerate(passages) if "needle" in passage
        )

        remaining_idxs = [idx for idx in range(len(passages)) if idx != evidence_idx]

        rank_idxs = [evidence_idx, *remaining_idxs]

        return [
            (idx, float(len(rank_idxs) - rank)) for rank, idx in enumerate(rank_idxs)
        ]

    monkeypatch.setattr(experiment, "rank_e5", fake_rank_e5)

    results = run_e5_full_index_validation(example, phase_chunks, model)

    assert tuple(results) == PHASES
    assert len(calls) == len(PHASES)

    for phase, call in zip(PHASES, calls):
        expected_passages = [chunk.chunk_text for chunk in phase_chunks[phase]]

        assert call["query"] == example["query"]
        assert call["model"] is model
        assert call["passages"] == expected_passages
        assert "needle" in call["passages"][results[phase]["ranking"][0][0]]
        assert results[phase]["ndcg_at_5"] == pytest.approx(1.0)
        assert results[phase]["oracle_normalized_coverage_at_5"] == pytest.approx(1.0)


def test_run_example_builds_rows_for_every_phase_and_retriever(monkeypatch):
    words = [f"word{i}" for i in range(600)]
    words[250] = "needle"
    doc = " ".join(words)

    evidence = " ".join(words[250:260])
    evidence_start = doc.index(evidence)

    example = {
        "query_id": "query-1",
        "document_id": "document-1",
        "query": "needle",
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    model = object()

    def fake_rank_e5(query, passages, recv_model):
        assert recv_model is model

        evidence_idx = next(
            idx for idx, passage in enumerate(passages) if "needle" in passage
        )

        remaining_idxs = [idx for idx in range(len(passages)) if idx != evidence_idx]

        rank_idxs = [evidence_idx, *remaining_idxs]

        return [
            (idx, float(len(rank_idxs) - rank)) for rank, idx in enumerate(rank_idxs)
        ]

    monkeypatch.setattr(experiment, "rank_e5", fake_rank_e5)

    results = run_example(example, model)

    assert len(results) == len(PHASES) * 2

    expected_pairs = [
        (phase, retriever) for phase in PHASES for retriever in ("bm25", "e5")
    ]

    got_pairs = [(row["phase"], row["retriever"]) for row in results]

    assert got_pairs == expected_pairs

    for row in results:
        assert row["query_id"] == "query-1"
        assert row["document_id"] == "document-1"
        assert row["chunk_size"] == CHUNK_SIZE
        assert row["overlap"] == OVERLAP
        assert row["retriever"] in {"bm25", "e5"}
        assert row["phase"] in PHASES
        assert row["top_k"] == TOP_K
        assert 0.0 <= row["efs"] <= 1.0
        assert row["rr"] == pytest.approx(1.0)
        assert row["ndcg_at_5"] == pytest.approx(1.0)
        assert row["oracle_normalized_coverage_at_5"] == pytest.approx(1.0)
        assert isinstance(row["best_chunk_index"], int)
        assert row["distractor_count"] == 2


def test_run_overlap_ablation_example_builds_reduced_overlap_rows(monkeypatch):
    words = [f"word{i}" for i in range(600)]
    words[250] = "needle"
    doc = " ".join(words)

    evidence = " ".join(words[250:260])
    evidence_start = doc.index(evidence)

    example = {
        "query_id": "query-1",
        "document_id": "document-1",
        "query": "needle",
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    model = object()

    def fake_rank_e5(query, passages, recv_model):
        assert recv_model is model

        evidence_idxs = [
            idx for idx, passage in enumerate(passages) if "needle" in passage
        ]
        remaining_idxs = [
            idx for idx in range(len(passages)) if idx not in evidence_idxs
        ]
        rank_idxs = [*evidence_idxs, *remaining_idxs]

        return [
            (idx, float(len(rank_idxs) - rank)) for rank, idx in enumerate(rank_idxs)
        ]

    monkeypatch.setattr(experiment, "rank_e5", fake_rank_e5)

    results = run_overlap_ablation_example(example, model)

    assert len(results) == len(OVERLAP_ABLATION_PHASES) * 2

    expected_pairs = [
        (phase, retriever)
        for phase in OVERLAP_ABLATION_PHASES
        for retriever in ("bm25", "e5")
    ]
    got_pairs = [(row["phase"], row["retriever"]) for row in results]

    assert got_pairs == expected_pairs

    for row in results:
        assert row["query_id"] == "query-1"
        assert row["document_id"] == "document-1"
        assert row["chunk_size"] == CHUNK_SIZE
        assert row["overlap"] == OVERLAP_ABLATION_OVERLAP
        assert row["phase"] in OVERLAP_ABLATION_PHASES
        assert row["retriever"] in {"bm25", "e5"}
        assert row["top_k"] == TOP_K
        assert 0.0 <= row["efs"] <= 1.0
        assert row["rr"] == pytest.approx(1.0)
        assert 0.0 <= row["ndcg_at_5"] <= 1.0

        if row["retriever"] == "e5":
            assert row["ndcg_at_5"] == pytest.approx(1.0)

        assert row["oracle_normalized_coverage_at_5"] == pytest.approx(1.0)
        assert isinstance(row["best_chunk_index"], int)
        assert row["distractor_count"] == 2


def test_run_dev_pilot_uses_five_distinct_documents(monkeypatch):
    train_examples = [
        {"query_id": "query-0", "document_id": "document-a"},
        {"query_id": "query-1", "document_id": "document-a"},
        {"query_id": "query-2", "document_id": "document-b"},
        {"query_id": "query-3", "document_id": "document-c"},
        {"query_id": "query-4", "document_id": "document-c"},
        {"query_id": "query-5", "document_id": "document-d"},
        {"query_id": "query-6", "document_id": "document-e"},
        {"query_id": "query-7", "document_id": "document-f"},
    ]

    model = object()
    seen_query_ids = []

    def fake_run_example(example, recv_model):
        assert recv_model is model
        seen_query_ids.append(example["query_id"])

        return [
            {"query_id": example["query_id"], "document_id": example["document_id"]}
        ]

    monkeypatch.setattr(experiment, "run_example", fake_run_example)

    results = run_dev_pilot(train_examples, model)

    assert seen_query_ids == ["query-0", "query-2", "query-3", "query-5", "query-6"]
    assert len(results) == DEV_PILOT_SIZE
    assert len({row["document_id"] for row in results}) == DEV_PILOT_SIZE


def test_run_dev_pilot_requires_five_distinct_documents():
    train_examples = [
        {"query_id": "query-0", "document_id": "document-a"},
        {"query_id": "query-1", "document_id": "document-a"},
        {"query_id": "query-2", "document_id": "document-b"},
        {"query_id": "query-3", "document_id": "document-c"},
        {"query_id": "query-4", "document_id": "document-d"},
    ]

    with pytest.raises(
        ValueError,
        match=("not enough distinct training documents for the development pilot"),
    ):
        run_dev_pilot(train_examples, model=object())


def test_run_overlap_ablation_uses_every_validation_example(monkeypatch):
    val_examples = [
        {"query_id": "query-0", "document_id": "document-a"},
        {"query_id": "query-1", "document_id": "document-a"},
        {"query_id": "query-2", "document_id": "document-b"},
    ]

    model = object()
    seen_query_ids = []

    def fake_run_overlap_example(example, recv_model):
        assert recv_model is model
        seen_query_ids.append(example["query_id"])

        return [
            {"query_id": example["query_id"], "document_id": example["document_id"]}
        ]

    monkeypatch.setattr(
        experiment, "run_overlap_ablation_example", fake_run_overlap_example
    )

    results = run_overlap_ablation(val_examples, model)

    assert seen_query_ids == ["query-0", "query-1", "query-2"]
    assert len(results) == 3


def test_run_primary_experiment_uses_every_validation_example(monkeypatch):
    val_examples = [
        {"query_id": "query-0", "document_id": "document-a"},
        {"query_id": "query-1", "document_id": "document-a"},
        {"query_id": "query-2", "document_id": "document-b"},
    ]

    model = object()
    seen_query_ids = []

    def fake_run_example(example, recv_model):
        assert recv_model is model
        seen_query_ids.append(example["query_id"])

        return [
            {"query_id": example["query_id"], "document_id": example["document_id"]}
        ]

    monkeypatch.setattr(experiment, "run_example", fake_run_example)

    results = run_primary_experiment(val_examples, model)

    assert seen_query_ids == ["query-0", "query-1", "query-2"]
    assert len(results) == 3
    assert [row["query_id"] for row in results] == ["query-0", "query-1", "query-2"]


def test_load_training_examples_uses_training_split_and_settings(monkeypatch, tmp_path):
    docs = {"document-1": {}}
    dialogues = [{"dial_id": "dialogue-1"}]
    expected_examples = [{"query_id": "query-1"}]

    seen = {}

    def fake_load_docs(path):
        seen["documents"] = path
        return docs

    def fake_load_dialogues(path):
        seen["dialogues"] = path
        return dialogues

    def fake_build_examples(recv_dialogues, recv_docs, chunk_size, phases):
        assert recv_dialogues is dialogues
        assert recv_docs is docs
        assert chunk_size == CHUNK_SIZE
        assert phases == PHASES
        return expected_examples

    monkeypatch.setattr(experiment, "load_docs", fake_load_docs)
    monkeypatch.setattr(experiment, "load_dialogues", fake_load_dialogues)
    monkeypatch.setattr(experiment, "build_examples", fake_build_examples)

    examples = load_training_examples(tmp_path)

    assert examples is expected_examples
    assert seen["documents"] == Path(tmp_path) / "doc2dial_doc.json"
    assert seen["dialogues"] == Path(tmp_path) / "doc2dial_dial_train.json"


def test_load_validation_examples_uses_validation_split_and_settings(
    monkeypatch, tmp_path
):
    docs = {"document-1": {}}
    dialogues = [{"dial_id": "dialogue-1"}]
    expected_examples = [{"query_id": "query-1"}]

    seen = {}

    def fake_load_docs(path):
        seen["documents"] = path
        return docs

    def fake_load_dialogues(path):
        seen["dialogues"] = path
        return dialogues

    def fake_build_examples(recv_dialogues, recv_docs, chunk_size, phases):
        assert recv_dialogues is dialogues
        assert recv_docs is docs
        assert chunk_size == CHUNK_SIZE
        assert phases == PHASES
        return expected_examples

    monkeypatch.setattr(experiment, "load_docs", fake_load_docs)
    monkeypatch.setattr(experiment, "load_dialogues", fake_load_dialogues)
    monkeypatch.setattr(experiment, "build_examples", fake_build_examples)

    examples = load_validation_examples(tmp_path)

    assert examples is expected_examples
    assert seen["documents"] == Path(tmp_path) / "doc2dial_doc.json"
    assert seen["dialogues"] == Path(tmp_path) / "doc2dial_dial_validation.json"


def test_run_dev_pilot_from_data_dir_loads_e5_once(monkeypatch, tmp_path):
    train_examples = [
        {"query_id": f"query-{idx}", "document_id": f"document-{idx}"}
        for idx in range(DEV_PILOT_SIZE)
    ]
    model = object()
    expected_results = [{"query_id": "query-0"}]

    load_count = 0

    def fake_load_train_examples(data_dir):
        assert data_dir == tmp_path
        return train_examples

    def fake_load_e5():
        nonlocal load_count
        load_count += 1
        return model

    def fake_run_dev_pilot(recv_examples, recv_model):
        assert recv_examples is train_examples
        assert recv_model is model
        return expected_results

    monkeypatch.setattr(experiment, "load_training_examples", fake_load_train_examples)
    monkeypatch.setattr(experiment, "load_e5", fake_load_e5)
    monkeypatch.setattr(experiment, "run_dev_pilot", fake_run_dev_pilot)

    results = run_dev_pilot_from_data_dir(tmp_path)

    assert results is expected_results
    assert load_count == 1


def test_run_primary_experiment_from_data_dir_loads_e5_once(monkeypatch, tmp_path):
    val_examples = [
        {"query_id": "query-0", "document_id": "document-a"},
        {"query_id": "query-1", "document_id": "document-b"},
    ]
    model = object()
    expected_results = [{"query_id": "query-0"}, {"query_id": "query-1"}]

    load_count = 0

    def fake_load_validation_examples(data_dir):
        assert data_dir == tmp_path
        return val_examples

    def fake_load_e5():
        nonlocal load_count
        load_count += 1
        return model

    def fake_run_primary(recv_examples, recv_model):
        assert recv_examples is val_examples
        assert recv_model is model
        return expected_results

    monkeypatch.setattr(
        experiment, "load_validation_examples", fake_load_validation_examples
    )
    monkeypatch.setattr(experiment, "load_e5", fake_load_e5)
    monkeypatch.setattr(experiment, "run_primary_experiment", fake_run_primary)

    results = run_primary_experiment_from_data_dir(tmp_path)

    assert results is expected_results
    assert load_count == 1


def test_run_overlap_ablation_from_data_dir_loads_e5_once(monkeypatch, tmp_path):
    val_examples = [
        {"query_id": "query-0", "document_id": "document-a"},
        {"query_id": "query-1", "document_id": "document-b"},
    ]
    model = object()
    expected_results = [{"query_id": "query-0"}, {"query_id": "query-1"}]
    load_count = 0

    def fake_load_validation_examples(data_dir):
        assert data_dir == tmp_path
        return val_examples

    def fake_load_e5():
        nonlocal load_count
        load_count += 1
        return model

    def fake_run_overlap(recv_examples, recv_model):
        assert recv_examples is val_examples
        assert recv_model is model
        return expected_results

    monkeypatch.setattr(
        experiment, "load_validation_examples", fake_load_validation_examples
    )
    monkeypatch.setattr(experiment, "load_e5", fake_load_e5)
    monkeypatch.setattr(experiment, "run_overlap_ablation", fake_run_overlap)

    results = run_overlap_ablation_from_data_dir(tmp_path)

    assert results is expected_results
    assert load_count == 1


def _make_valid_result_table(phases=PHASES, overlap=OVERLAP):
    return [
        {
            "query_id": "query-1",
            "document_id": "document-1",
            "chunk_size": CHUNK_SIZE,
            "overlap": overlap,
            "phase": phase,
            "retriever": retriever,
            "top_k": TOP_K,
            "efs": 0.25,
            "rr": 0.5,
            "ndcg_at_5": 0.75,
            "oracle_normalized_coverage_at_5": 1.0,
            "best_chunk_index": 1,
            "distractor_count": 3,
        }
        for phase in phases
        for retriever in ("bm25", "e5")
    ]


def test_validate_overlap_ablation_results_accepts_complete_table():
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]

    results = _make_valid_result_table(
        OVERLAP_ABLATION_PHASES, OVERLAP_ABLATION_OVERLAP
    )

    validate_overlap_ablation_results(results, val_examples)


def test_validate_overlap_ablation_results_rejects_missing_condition():
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]

    results = _make_valid_result_table(
        OVERLAP_ABLATION_PHASES, OVERLAP_ABLATION_OVERLAP
    )
    results.pop()

    with pytest.raises(ValueError, match="unexpected overlap ablation result count"):
        validate_overlap_ablation_results(results, val_examples)


def test_save_overlap_ablation_results_writes_expected_csv(tmp_path):
    results = [
        {
            "query_id": "query-1",
            "document_id": "document-1",
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP_ABLATION_OVERLAP,
            "phase": 0,
            "retriever": "bm25",
            "top_k": TOP_K,
            "efs": 0.0,
            "rr": 1.0,
            "ndcg_at_5": 1.0,
            "oracle_normalized_coverage_at_5": 1.0,
            "best_chunk_index": 2,
            "distractor_count": 4,
        }
    ]
    out_path = tmp_path / "results" / "overlap_ablation_results.csv"

    saved = save_overlap_ablation_results(results, out_path)

    assert saved == out_path
    assert out_path.exists()

    with out_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert tuple(rows[0]) == RESULT_FIELDS
    assert rows[0]["overlap"] == "40"
    assert rows[0]["phase"] == "0"


def test_run_and_save_overlap_ablation_validates_before_saving(monkeypatch, tmp_path):
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]
    model = object()
    results = [{"query_id": "query-1"}]
    out_path = tmp_path / "results" / "overlap_ablation_results.csv"
    calls = []

    def fake_load_validation_examples(data_dir):
        assert data_dir == tmp_path
        calls.append("load_examples")
        return val_examples

    def fake_load_e5():
        calls.append("load_model")
        return model

    def fake_run_overlap(recv_examples, recv_model):
        assert recv_examples is val_examples
        assert recv_model is model
        calls.append("run")
        return results

    def fake_validate_overlap_results(recv_results, recv_examples):
        assert recv_results is results
        assert recv_examples is val_examples
        calls.append("validate")

    def fake_save_overlap_results(recv_results, recv_out_path):
        assert recv_results is results
        assert recv_out_path == out_path
        calls.append("save")
        return out_path

    monkeypatch.setattr(
        experiment, "load_validation_examples", fake_load_validation_examples
    )
    monkeypatch.setattr(experiment, "load_e5", fake_load_e5)
    monkeypatch.setattr(experiment, "run_overlap_ablation", fake_run_overlap)
    monkeypatch.setattr(
        experiment, "validate_overlap_ablation_results", fake_validate_overlap_results
    )
    monkeypatch.setattr(
        experiment, "save_overlap_ablation_results", fake_save_overlap_results
    )

    saved = run_and_save_overlap_ablation(tmp_path, out_path)

    assert saved == out_path
    assert calls == ["load_examples", "load_model", "run", "validate", "save"]


def test_run_and_save_overlap_ablation_rejects_existing_output(monkeypatch, tmp_path):
    out_path = tmp_path / "overlap_ablation_results.csv"
    out_path.write_text("existing results\n")

    def fail_call(*args, **kwargs):
        pytest.fail("ablation should not run when output already exists")

    monkeypatch.setattr(experiment, "load_validation_examples", fail_call)
    monkeypatch.setattr(experiment, "load_e5", fail_call)

    with pytest.raises(FileExistsError, match="overlap ablation results already exist"):
        run_and_save_overlap_ablation(tmp_path, out_path)


def test_validate_primary_results_accepts_complete_result_table():
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]

    results = _make_valid_result_table()

    validate_primary_results(results, val_examples)


def test_validate_primary_results_rejects_cross_retriever_efs_mismatch():
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]
    results = _make_valid_result_table()

    e5_row = next(
        row for row in results if row["phase"] == 0 and row["retriever"] == "e5"
    )
    e5_row["efs"] = 0.2

    with pytest.raises(ValueError, match="EFS differs between retrievers"):
        validate_primary_results(results, val_examples)


def test_validate_primary_results_rejects_cross_retriever_best_chunk_mismatch():
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]
    results = _make_valid_result_table()

    e5_row = next(
        row for row in results if row["phase"] == 0 and row["retriever"] == "e5"
    )
    e5_row["best_chunk_index"] = 2

    with pytest.raises(ValueError, match="best_chunk_index differs between retrievers"):
        validate_primary_results(results, val_examples)


def test_validate_primary_results_rejects_changing_distractor_count():
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]
    results = _make_valid_result_table()

    bm25_row = next(
        row for row in results if row["phase"] == 25 and row["retriever"] == "bm25"
    )
    bm25_row["distractor_count"] = 4

    with pytest.raises(ValueError, match="distractor_count changes across conditions"):
        validate_primary_results(results, val_examples)


def test_validate_primary_results_rejects_missing_condition():
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]

    results = _make_valid_result_table()

    results.pop()

    with pytest.raises(ValueError, match="unexpected primary result count"):
        validate_primary_results(results, val_examples)


def test_save_primary_results_writes_expected_csv(tmp_path):
    results = [
        {
            "query_id": "query-1",
            "document_id": "document-1",
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "phase": 0,
            "retriever": "bm25",
            "top_k": TOP_K,
            "efs": 0.0,
            "rr": 1.0,
            "ndcg_at_5": 1.0,
            "oracle_normalized_coverage_at_5": 1.0,
            "best_chunk_index": 2,
            "distractor_count": 4,
        }
    ]

    out_path = tmp_path / "results" / "primary_results.csv"

    saved = save_primary_results(results, out_path)

    assert saved == out_path
    assert out_path.exists()

    with out_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert tuple(rows[0]) == RESULT_FIELDS
    assert rows[0]["query_id"] == "query-1"
    assert rows[0]["document_id"] == "document-1"
    assert rows[0]["chunk_size"] == "200"
    assert rows[0]["overlap"] == "0"
    assert rows[0]["phase"] == "0"
    assert rows[0]["retriever"] == "bm25"
    assert rows[0]["top_k"] == "5"


def test_run_and_save_primary_experiment_validates_before_saving(monkeypatch, tmp_path):
    val_examples = [{"query_id": "query-1", "document_id": "document-1"}]
    model = object()
    results = [{"query_id": "query-1"}]
    out_path = tmp_path / "results" / "primary_results.csv"
    calls = []

    def fake_load_validation_examples(data_dir):
        assert data_dir == tmp_path
        calls.append("load_examples")
        return val_examples

    def fake_load_e5():
        calls.append("load_model")
        return model

    def fake_run_primary(recv_examples, recv_model):
        assert recv_examples is val_examples
        assert recv_model is model
        calls.append("run")
        return results

    def fake_validate_primary_results(recv_results, recv_examples):
        assert recv_results is results
        assert recv_examples is val_examples
        calls.append("validate")

    def fake_save_primary_results(recv_results, recv_out_path):
        assert recv_results is results
        assert recv_out_path == out_path
        calls.append("save")
        return out_path

    monkeypatch.setattr(
        experiment, "load_validation_examples", fake_load_validation_examples
    )
    monkeypatch.setattr(experiment, "load_e5", fake_load_e5)
    monkeypatch.setattr(experiment, "run_primary_experiment", fake_run_primary)
    monkeypatch.setattr(
        experiment, "validate_primary_results", fake_validate_primary_results
    )
    monkeypatch.setattr(experiment, "save_primary_results", fake_save_primary_results)

    saved = run_and_save_primary_experiment(tmp_path, out_path)

    assert saved == out_path
    assert calls == ["load_examples", "load_model", "run", "validate", "save"]


def test_run_and_save_primary_experiment_rejects_existing_output(monkeypatch, tmp_path):
    out_path = tmp_path / "primary_results.csv"
    out_path.write_text("existing results\n")

    def fail_call(*args, **kwargs):
        pytest.fail("experiment should not run when output already exists")

    monkeypatch.setattr(experiment, "load_validation_examples", fail_call)
    monkeypatch.setattr(experiment, "load_e5", fail_call)

    with pytest.raises(FileExistsError, match="primary results already exist"):
        run_and_save_primary_experiment(tmp_path, out_path)


def test_generate_efs_rr_figure_creates_output_file(tmp_path):
    results = [
        {"retriever": "bm25", "efs": 0.0, "rr": 1.0},
        {"retriever": "bm25", "efs": 0.5, "rr": 0.5},
        {"retriever": "e5", "efs": 0.0, "rr": 1.0},
        {"retriever": "e5", "efs": 0.5, "rr": 0.75},
    ]
    out_path = tmp_path / "figures" / "efs_vs_rr.png"

    saved = gen_efs_rr_figure(results, out_path)

    assert saved == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_generate_efs_rr_figure_requires_both_retrievers(tmp_path):
    results = [{"retriever": "bm25", "efs": 0.0, "rr": 1.0}]

    with pytest.raises(ValueError, match="no results found for retriever: e5"):
        gen_efs_rr_figure(results, tmp_path / "efs_vs_rr.png")


def test_build_error_case_records_reconstructs_selected_phases(monkeypatch):
    words = [f"word{i}" for i in range(500)]
    doc = " ".join(words)
    evidence = " ".join(words[195:205])
    evidence_start = doc.index(evidence)
    example = {
        "query_id": "query-1",
        "document_id": "document-1",
        "query": "what is the evidence?",
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }

    phase_info = build_phase_evidence_info(example, build_phase_chunks(example))

    selected = {
        "failures": [
            {
                "query_id": "query-1",
                "document_id": "document-1",
                "low_efs": phase_info[25]["efs"],
                "high_efs": phase_info[0]["efs"],
                "low_phases": (25,),
                "high_phases": (0,),
                "bm25_rr_difference": -0.5,
                "e5_rr_difference": -0.25,
                "focus_retriever": "bm25",
            }
        ],
        "robust": [],
        "disagreements": [],
    }

    monkeypatch.setattr(
        experiment, "select_error_cases", lambda results, n_cases: selected
    )

    results = []

    for phase, bm25_rr, e5_rr in ((25, 1.0, 0.75), (0, 0.5, 0.5)):
        for retriever, rr in (("bm25", bm25_rr), ("e5", e5_rr)):
            results.append(
                {
                    "query_id": "query-1",
                    "document_id": "document-1",
                    "phase": phase,
                    "retriever": retriever,
                    "efs": phase_info[phase]["efs"],
                    "rr": rr,
                    "best_chunk_index": phase_info[phase]["best_chunk_index"],
                }
            )

    records = build_error_case_records(results, [example], n_cases=1)

    record = records["failures"][0]

    assert record["query"] == "what is the evidence?"
    assert record["evidence_text"] == evidence
    assert record["focus_retriever"] == "bm25"
    assert record["low_conditions"][0]["phase"] == 25
    assert record["low_conditions"][0]["efs"] == 0.0
    assert record["low_conditions"][0]["bm25_rr"] == 1.0
    assert evidence in record["low_conditions"][0]["passage"]
    assert record["high_conditions"][0]["phase"] == 0
    assert record["high_conditions"][0]["efs"] == pytest.approx(0.5)
    assert record["high_conditions"][0]["bm25_rr"] == 0.5
    assert record["high_conditions"][0]["evidence_overlap_fraction"] == pytest.approx(
        0.5
    )


def test_build_error_case_records_requires_selected_validation_example(monkeypatch):
    selected = {
        "failures": [{"query_id": "missing-query", "document_id": "document-1"}],
        "robust": [],
        "disagreements": [],
    }

    monkeypatch.setattr(
        experiment, "select_error_cases", lambda results, n_cases: selected
    )

    with pytest.raises(
        ValueError, match="validation example not found for query: missing-query"
    ):
        build_error_case_records([], [], n_cases=1)


def test_build_error_case_records_checks_reconstructed_result_metadata(monkeypatch):
    words = [f"word{i}" for i in range(500)]
    doc = " ".join(words)
    evidence = " ".join(words[195:205])
    evidence_start = doc.index(evidence)
    example = {
        "query_id": "query-1",
        "document_id": "document-1",
        "query": "query",
        "document_text": doc,
        "evidence_start": evidence_start,
        "evidence_end": evidence_start + len(evidence),
    }
    phase_info = build_phase_evidence_info(example, build_phase_chunks(example))
    selected = {
        "failures": [
            {
                "query_id": "query-1",
                "document_id": "document-1",
                "low_efs": 0.0,
                "high_efs": phase_info[0]["efs"],
                "low_phases": (25,),
                "high_phases": (0,),
                "bm25_rr_difference": -0.5,
                "e5_rr_difference": -0.25,
                "focus_retriever": "bm25",
            }
        ],
        "robust": [],
        "disagreements": [],
    }

    monkeypatch.setattr(
        experiment, "select_error_cases", lambda results, n_cases: selected
    )

    results = []

    for phase in (25, 0):
        for retriever in ("bm25", "e5"):
            results.append(
                {
                    "query_id": "query-1",
                    "document_id": "document-1",
                    "phase": phase,
                    "retriever": retriever,
                    "efs": phase_info[phase]["efs"],
                    "rr": 0.5,
                    "best_chunk_index": phase_info[phase]["best_chunk_index"],
                }
            )

    results[0]["best_chunk_index"] += 1

    with pytest.raises(
        ValueError, match="stored best chunk does not match reconstructed phase"
    ):
        build_error_case_records(results, [example], n_cases=1)


def test_save_error_case_records_writes_json(tmp_path):
    records = {
        "failures": [{"query_id": "query-1", "query": "café"}],
        "robust": [],
        "disagreements": [],
    }
    out_path = tmp_path / "results" / "error_cases.json"

    saved = save_error_case_records(records, out_path)

    assert saved == out_path
    assert json.loads(out_path.read_text(encoding="utf-8")) == records
    assert "café" in out_path.read_text(encoding="utf-8")


def test_main_runs_primary_experiment(monkeypatch, tmp_path, capsys):
    data_dir = tmp_path / "doc2dial"
    out_path = tmp_path / "primary.csv"
    calls = {}

    def fake_run_save(data_path, results_path):
        calls["data_dir"] = data_path
        calls["output_path"] = results_path
        return Path(results_path)

    monkeypatch.setattr(experiment, "run_and_save_primary_experiment", fake_run_save)

    code = experiment.main(["primary", str(data_dir), str(out_path)])

    assert code == 0
    assert calls == {"data_dir": data_dir, "output_path": out_path}
    assert capsys.readouterr().out == f"Saved results: {out_path}\n"


def test_main_runs_overlap_ablation(monkeypatch, tmp_path, capsys):
    data_dir = tmp_path / "doc2dial"
    out_path = tmp_path / "overlap.csv"
    calls = {}

    def fake_run_save(data_path, results_path):
        calls["data_dir"] = data_path
        calls["output_path"] = results_path
        return Path(results_path)

    monkeypatch.setattr(experiment, "run_and_save_overlap_ablation", fake_run_save)

    code = experiment.main(["overlap", str(data_dir), str(out_path)])

    assert code == 0
    assert calls == {"data_dir": data_dir, "output_path": out_path}
    assert capsys.readouterr().out == f"Saved results: {out_path}\n"
