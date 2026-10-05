import json

import pytest

from boundary_retrieval.data import (
    away_from_clipped_edges,
    build_dataset_summary,
    build_examples,
    extract_first_exchange,
    get_evidence_word_bounds,
    has_fragmentable_evidence,
    load_dialogues,
    load_docs,
    resolve_evidence,
)


def test_load_documents_indexes_documents_by_id(tmp_path) -> None:
    path = tmp_path / "doc2dial_doc.json"
    data = {
        "doc_data": {
            "ssa": {
                "doc-a": {
                    "doc_id": "doc-a",
                    "domain": "ssa",
                    "doc_text": "Alpha beta gamma.",
                    "spans": {
                        "1": {
                            "id_sp": "1",
                            "start_sp": 0,
                            "end_sp": 5,
                            "text_sp": "Alpha",
                        }
                    },
                }
            },
            "va": {
                "doc-b": {
                    "doc_id": "doc-b",
                    "domain": "va",
                    "doc_text": "Delta epsilon.",
                    "spans": {},
                }
            },
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    docs = load_docs(path)

    assert list(docs) == ["doc-a", "doc-b"]
    assert docs["doc-a"]["doc_text"] == "Alpha beta gamma."
    assert docs["doc-a"]["spans"]["1"]["text_sp"] == "Alpha"


def test_load_dialogues_returns_split_dialogues(tmp_path) -> None:
    path = tmp_path / "doc2dial_dial_train.json"
    data = {
        "dial_data": {
            "ssa": {
                "doc-a": [
                    {
                        "dial_id": "dial-1",
                        "doc_id": "doc-a",
                        "domain": "ssa",
                        "turns": [
                            {
                                "turn_id": 1,
                                "role": "user",
                                "utterance": "Question?",
                                "references": [],
                            }
                        ],
                    }
                ]
            }
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    dialogues = load_dialogues(path)

    assert len(dialogues) == 1
    assert dialogues[0]["dial_id"] == "dial-1"
    assert dialogues[0]["doc_id"] == "doc-a"
    assert dialogues[0]["turns"][0]["role"] == "user"


def test_extract_first_exchange_uses_first_user_turn() -> None:
    dialogue = {
        "dial_id": "dial-1",
        "doc_id": "doc-a",
        "turns": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "How do I apply?",
                "references": [],
            },
            {
                "turn_id": 2,
                "role": "agent",
                "utterance": "You can apply online.",
                "references": [{"sp_id": "10", "label": "solution"}],
            },
            {
                "turn_id": 3,
                "role": "user",
                "utterance": "What happens next?",
                "references": [],
            },
            {
                "turn_id": 4,
                "role": "agent",
                "utterance": "Next step.",
                "references": [{"sp_id": "20", "label": "solution"}],
            },
        ],
    }

    exchange = extract_first_exchange(dialogue)

    assert exchange == {
        "query_id": "dial-1_1",
        "document_id": "doc-a",
        "query": "How do I apply?",
        "references": [{"sp_id": "10", "label": "solution"}],
    }


def test_extract_first_exchange_keeps_grounding_labels() -> None:
    dialogue = {
        "dial_id": "dial-2",
        "doc_id": "doc-b",
        "turns": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "Am I eligible?",
                "references": [],
            },
            {
                "turn_id": 2,
                "role": "agent",
                "utterance": "Eligibility depends on these conditions.",
                "references": [
                    {"sp_id": "4", "label": "precondition"},
                    {"sp_id": "5", "label": "solution"},
                    {"sp_id": "6", "label": "other"},
                ],
            },
        ],
    }

    exchange = extract_first_exchange(dialogue)

    assert exchange is not None
    assert exchange["references"] == [
        {"sp_id": "4", "label": "precondition"},
        {"sp_id": "5", "label": "solution"},
    ]


def test_extract_first_exchange_does_not_use_later_grounding() -> None:
    dialogue = {
        "dial_id": "dial-3",
        "doc_id": "doc-c",
        "turns": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "First question",
                "references": [],
            },
            {
                "turn_id": 2,
                "role": "agent",
                "utterance": "No document grounding.",
                "references": [],
            },
            {
                "turn_id": 3,
                "role": "user",
                "utterance": "Second question",
                "references": [],
            },
            {
                "turn_id": 4,
                "role": "agent",
                "utterance": "Grounded answer.",
                "references": [{"sp_id": "8", "label": "solution"}],
            },
        ],
    }

    assert extract_first_exchange(dialogue) is None


def test_resolve_evidence_uses_document_offsets() -> None:
    exchange = {
        "query_id": "dial-1_1",
        "document_id": "doc-a",
        "query": "What comes after alpha?",
        "references": [{"sp_id": "2", "label": "solution"}],
    }
    doc = {
        "doc_text": "Alpha beta gamma.",
        "spans": {"2": {"id_sp": "2", "start_sp": 6, "end_sp": 10, "text_sp": "beta"}},
    }

    example = resolve_evidence(exchange, doc)

    assert example == {
        "query_id": "dial-1_1",
        "document_id": "doc-a",
        "query": "What comes after alpha?",
        "document_text": "Alpha beta gamma.",
        "evidence_start": 6,
        "evidence_end": 10,
    }


def test_resolve_evidence_combines_contiguous_spans() -> None:
    exchange = {
        "query_id": "dial-2_1",
        "document_id": "doc-b",
        "query": "What is the phrase?",
        "references": [
            {"sp_id": "2", "label": "solution"},
            {"sp_id": "1", "label": "precondition"},
        ],
    }
    doc = {
        "doc_text": "Alpha beta gamma.",
        "spans": {
            "1": {"id_sp": "1", "start_sp": 0, "end_sp": 6, "text_sp": "Alpha "},
            "2": {"id_sp": "2", "start_sp": 6, "end_sp": 10, "text_sp": "beta"},
        },
    }

    example = resolve_evidence(exchange, doc)

    assert example is not None
    assert example["evidence_start"] == 0
    assert example["evidence_end"] == 10
    assert (
        example["document_text"][example["evidence_start"] : example["evidence_end"]]
        == "Alpha beta"
    )


def test_resolve_evidence_rejects_separate_spans() -> None:
    exchange = {
        "query_id": "dial-3_1",
        "document_id": "doc-c",
        "query": "Question?",
        "references": [
            {"sp_id": "1", "label": "precondition"},
            {"sp_id": "2", "label": "solution"},
        ],
    }
    doc = {
        "doc_text": "Alpha beta gamma.",
        "spans": {
            "1": {"id_sp": "1", "start_sp": 0, "end_sp": 5, "text_sp": "Alpha"},
            "2": {"id_sp": "2", "start_sp": 11, "end_sp": 16, "text_sp": "gamma"},
        },
    }

    assert resolve_evidence(exchange, doc) is None


def test_resolve_evidence_detects_bad_span_text() -> None:
    exchange = {
        "query_id": "dial-4_1",
        "document_id": "doc-d",
        "query": "Question?",
        "references": [{"sp_id": "1", "label": "solution"}],
    }
    doc = {
        "doc_text": "Alpha beta.",
        "spans": {"1": {"id_sp": "1", "start_sp": 0, "end_sp": 5, "text_sp": "Wrong"}},
    }

    with pytest.raises(ValueError, match="does not match the document text"):
        resolve_evidence(exchange, doc)


def test_get_evidence_word_bounds_uses_source_words() -> None:
    doc_text = "zero  one\ntwo\tthree four"
    evidence_start = doc_text.index("one")
    evidence_end = doc_text.index("three") + len("three")

    example = {
        "document_text": doc_text,
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
    }

    assert get_evidence_word_bounds(example) == (1, 4)


def test_get_evidence_word_bounds_rejects_wordless_span() -> None:
    doc_text = "alpha   beta"
    example = {"document_text": doc_text, "evidence_start": 5, "evidence_end": 8}

    with pytest.raises(ValueError, match="does not contain any source words"):
        get_evidence_word_bounds(example)


def _make_word_example(
    evidence_start_word: int, evidence_end_word: int, word_count: int = 500
) -> dict:
    words = [f"w{idx}" for idx in range(word_count)]
    doc_text = " ".join(words)

    if evidence_start_word == 0:
        evidence_start = 0
    else:
        evidence_start = len(" ".join(words[:evidence_start_word])) + 1

    evidence_end = len(" ".join(words[:evidence_end_word]))

    return {
        "document_text": doc_text,
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
    }


def test_has_fragmentable_evidence_accepts_span_crossing_phase_boundary() -> None:
    example = _make_word_example(20, 30)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    assert has_fragmentable_evidence(example, 200, phases)


def test_has_fragmentable_evidence_rejects_chunk_sized_evidence() -> None:
    example = _make_word_example(20, 220)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    assert not has_fragmentable_evidence(example, 200, phases)


def test_has_fragmentable_evidence_rejects_unaffected_span() -> None:
    example = _make_word_example(1, 10)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    assert not has_fragmentable_evidence(example, 200, phases)


def test_boundary_at_evidence_edge_does_not_count_as_fragmentation() -> None:
    example = _make_word_example(25, 30)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    assert not has_fragmentable_evidence(example, 200, phases)


def test_edge_check_accepts_interior_evidence() -> None:
    example = _make_word_example(220, 230, word_count=500)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    assert away_from_clipped_edges(example, 200, phases)


def test_edge_check_rejects_evidence_in_shifted_prefix() -> None:
    example = _make_word_example(170, 180, word_count=500)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    assert not away_from_clipped_edges(example, 200, phases)


def test_edge_check_rejects_evidence_in_clipped_suffix() -> None:
    example = _make_word_example(320, 330, word_count=500)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    assert not away_from_clipped_edges(example, 200, phases)


def test_edge_check_allows_evidence_ending_at_clipped_suffix() -> None:
    example = _make_word_example(315, 325, word_count=500)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    assert away_from_clipped_edges(example, 200, phases)


def _make_grounded_dialogue(
    evidence_start_word: int,
    evidence_end_word: int,
    *,
    dialogue_id: str = "dial-1",
    doc_id: str = "doc-a",
    word_count: int = 500,
) -> tuple[dict, dict]:
    example = _make_word_example(evidence_start_word, evidence_end_word, word_count)
    doc_text = example["document_text"]
    evidence_start = example["evidence_start"]
    evidence_end = example["evidence_end"]
    evidence_text = doc_text[evidence_start:evidence_end]

    doc = {
        "doc_id": doc_id,
        "doc_text": doc_text,
        "spans": {
            "1": {
                "id_sp": "1",
                "start_sp": evidence_start,
                "end_sp": evidence_end,
                "text_sp": evidence_text,
            }
        },
    }

    dialogue = {
        "dial_id": dialogue_id,
        "doc_id": doc_id,
        "turns": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "Where is the evidence?",
                "references": [],
            },
            {
                "turn_id": 2,
                "role": "agent",
                "utterance": "Here it is.",
                "references": [{"sp_id": "1", "label": "solution"}],
            },
        ],
    }

    return dialogue, doc


def test_build_examples_keeps_qualifying_dialogue() -> None:
    dialogue, doc = _make_grounded_dialogue(220, 230)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    examples = build_examples([dialogue], {"doc-a": doc}, chunk_size=200, phases=phases)

    assert len(examples) == 1

    example = examples[0]

    assert example["query_id"] == "dial-1_1"
    assert example["document_id"] == "doc-a"
    assert example["query"] == "Where is the evidence?"

    evidence_text = example["document_text"][
        example["evidence_start"] : example["evidence_end"]
    ]

    assert evidence_text.startswith("w220")
    assert evidence_text.endswith("w229")


def test_build_examples_filters_unaffected_evidence() -> None:
    dialogue, doc = _make_grounded_dialogue(205, 215)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    examples = build_examples([dialogue], {"doc-a": doc}, chunk_size=200, phases=phases)

    assert examples == []


def test_build_examples_filters_edge_evidence() -> None:
    dialogue, doc = _make_grounded_dialogue(170, 180)
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    examples = build_examples([dialogue], {"doc-a": doc}, chunk_size=200, phases=phases)

    assert examples == []


def test_build_dataset_summary_reports_split_counts_and_lengths() -> None:
    train_one = _make_word_example(1, 3, word_count=5)
    train_one["document_id"] = "doc-a"

    train_two = _make_word_example(2, 4, word_count=5)
    train_two["document_id"] = "doc-a"

    value = _make_word_example(1, 4, word_count=6)
    value["document_id"] = "doc-b"

    summary = build_dataset_summary(
        [train_one, train_two],
        [value],
        src_version="1.0.1",
        archive_sha="abc123",
        chunk_size=200,
        phases=(0, 25, 50),
    )

    assert summary["source"] == {
        "dataset": "Doc2Dial",
        "version": "1.0.1",
        "archive_sha256": "abc123",
    }

    assert summary["settings"] == {"chunk_size_words": 200, "phases_words": [0, 25, 50]}

    assert summary["splits"]["train"]["examples"] == 2
    assert summary["splits"]["train"]["documents"] == 1
    assert summary["splits"]["train"]["document_words"] == {
        "min": 5,
        "median": 5,
        "mean": 5,
        "max": 5,
    }
    assert summary["splits"]["train"]["evidence_words"] == {
        "min": 2,
        "median": 2.0,
        "mean": 2,
        "max": 2,
    }

    assert summary["splits"]["validation"]["examples"] == 1
    assert summary["splits"]["validation"]["documents"] == 1
    assert summary["splits"]["validation"]["document_words"]["mean"] == 6
    assert summary["splits"]["validation"]["evidence_words"]["mean"] == 3


def test_build_dataset_summary_rejects_empty_split() -> None:
    value = _make_word_example(1, 3, word_count=5)
    value["document_id"] = "doc-b"

    with pytest.raises(ValueError, match="empty example split"):
        build_dataset_summary(
            [],
            [value],
            src_version="1.0.1",
            archive_sha="abc123",
            chunk_size=200,
            phases=(0, 25),
        )
