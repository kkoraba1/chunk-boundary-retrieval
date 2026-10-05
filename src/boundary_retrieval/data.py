import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_docs(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load Doc2Dial documents into a dict keyed by document ID."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    docs = {}

    for domain_docs in data["doc_data"].values():
        for doc_id, doc in domain_docs.items():
            docs[doc_id] = doc

    return docs


def load_dialogues(path: str | Path) -> list[dict[str, Any]]:
    """Load all dialogues from one Doc2Dial split."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    dialogues = []

    for domain_dialogues in data["dial_data"].values():
        for doc_dialogues in domain_dialogues.values():
            dialogues.extend(doc_dialogues)

    return dialogues


def extract_first_exchange(dialogue: dict[str, Any]) -> dict[str, Any] | None:
    """Get the first user query and the next agent turn's evidence refs."""
    turns = dialogue["turns"]

    for idx, turn in enumerate(turns):
        if turn["role"] != "user":
            continue

        if idx + 1 >= len(turns):
            return None

        agent_turn = turns[idx + 1]

        if agent_turn["role"] != "agent":
            return None

        refs = [
            ref
            for ref in agent_turn.get("references", [])
            if ref.get("label") in {"solution", "precondition"}
        ]

        if not refs:
            return None

        return {
            "query_id": f"{dialogue['dial_id']}_{turn['turn_id']}",
            "document_id": dialogue["doc_id"],
            "query": turn["utterance"],
            "references": refs,
        }

    return None


def resolve_evidence(
    exchange: dict[str, Any], doc: dict[str, Any]
) -> dict[str, Any] | None:
    """Return one evidence span, or None if the refs leave a gap."""
    doc_text = doc["doc_text"]
    spans = []

    for ref in exchange["references"]:
        span = doc["spans"][ref["sp_id"]]
        start = span["start_sp"]
        end = span["end_sp"]

        if doc_text[start:end] != span["text_sp"]:
            raise ValueError(f"Span {ref['sp_id']} does not match the document text")

        spans.append((start, end))

    spans.sort()

    evidence_start, evidence_end = spans[0]

    for start, end in spans[1:]:
        if start > evidence_end:
            return None

        evidence_end = max(evidence_end, end)

    return {
        "query_id": exchange["query_id"],
        "document_id": exchange["document_id"],
        "query": exchange["query"],
        "document_text": doc_text,
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
    }


def get_evidence_word_bounds(example: dict[str, Any]) -> tuple[int, int]:
    """Return the evidence's word range, with an exclusive end."""
    doc_text = example["document_text"]
    evidence_start = example["evidence_start"]
    evidence_end = example["evidence_end"]

    words = list(re.finditer(r"\S+", doc_text))
    evidence_words = [
        idx
        for idx, match in enumerate(words)
        if match.start() < evidence_end and match.end() > evidence_start
    ]

    if not evidence_words:
        raise ValueError("Evidence span does not contain any source words")

    return evidence_words[0], evidence_words[-1] + 1


def has_fragmentable_evidence(
    example: dict[str, Any], chunk_size: int, phases: tuple[int, ...]
) -> bool:
    """Check if evidence is shorter than a chunk and split by any given phase."""
    evidence_start, evidence_end = get_evidence_word_bounds(example)

    if evidence_end - evidence_start >= chunk_size:
        return False

    for phase in phases:
        if phase < 0 or phase >= chunk_size:
            raise ValueError("phase must be between 0 and chunk_size - 1")

        first_boundary = chunk_size if phase == 0 else phase

        for boundary in range(first_boundary, evidence_end, chunk_size):
            if evidence_start < boundary < evidence_end:
                return True

    return False


def away_from_clipped_edges(
    example: dict[str, Any], chunk_size: int, phases: tuple[int, ...]
) -> bool:
    """Check that evidence avoids shortened edge chunks in every phase."""
    evidence_start, evidence_end = get_evidence_word_bounds(example)
    word_count = len(re.findall(r"\S+", example["document_text"]))

    for phase in phases:
        if phase < 0 or phase >= chunk_size:
            raise ValueError("phase must be between 0 and chunk_size - 1")

    active = [phase for phase in phases if 0 < phase < word_count]

    if active and evidence_start < max(active):
        return False

    clipped_suffix_starts = []

    for phase in phases:
        if phase >= word_count:
            continue

        remaining_words = word_count - phase
        remainder = remaining_words % chunk_size

        if remainder != 0:
            clipped_suffix_starts.append(word_count - remainder)

    return not clipped_suffix_starts or evidence_end <= min(clipped_suffix_starts)


def build_examples(
    dialogues: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    chunk_size: int,
    phases: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Build filtered retrieval examples from one dialogue split."""
    examples = []

    for dialogue in dialogues:
        exchange = extract_first_exchange(dialogue)

        if exchange is None:
            continue

        doc = docs[exchange["document_id"]]
        example = resolve_evidence(exchange, doc)

        if example is None:
            continue

        if not has_fragmentable_evidence(example, chunk_size, phases):
            continue

        if not away_from_clipped_edges(example, chunk_size, phases):
            continue

        examples.append(example)

    return examples


def _summarize_lengths(lengths: list[int]) -> dict[str, int | float]:
    return {
        "min": min(lengths),
        "median": median(lengths),
        "mean": round(mean(lengths), 2),
        "max": max(lengths),
    }


def _summarize_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot summarize an empty example split")

    docs = {}
    evidence_lengths = []

    for example in examples:
        docs[example["document_id"]] = example["document_text"]

        evidence_start, evidence_end = get_evidence_word_bounds(example)
        evidence_lengths.append(evidence_end - evidence_start)

    doc_lengths = [len(re.findall(r"\S+", doc_text)) for doc_text in docs.values()]

    return {
        "examples": len(examples),
        "documents": len(docs),
        "document_words": _summarize_lengths(doc_lengths),
        "evidence_words": _summarize_lengths(evidence_lengths),
    }


def build_dataset_summary(
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    *,
    src_version: str,
    archive_sha: str,
    chunk_size: int,
    phases: tuple[int, ...],
) -> dict[str, Any]:
    """Summarize the data source, chunk settings, and filtered splits."""
    return {
        "source": {
            "dataset": "Doc2Dial",
            "version": src_version,
            "archive_sha256": archive_sha,
        },
        "settings": {"chunk_size_words": chunk_size, "phases_words": list(phases)},
        "splits": {
            "train": _summarize_examples(train_examples),
            "validation": _summarize_examples(val_examples),
        },
    }
