import numpy as np
import pytest

from boundary_retrieval import retrieval
from boundary_retrieval.retrieval import (
    E5_MAX_TOKENS,
    bm25_tokens,
    check_e5_length,
    fmt_e5_passage,
    fmt_e5_query,
    rank_bm25,
    rank_e5,
)


def test_tokenize_bm25_lowercases_and_uses_regex_words() -> None:
    text = "Alpha, BETA... gamma-42!"

    assert bm25_tokens(text) == ["alpha", "beta", "gamma", "42"]


def test_rank_with_bm25_places_matching_passage_first() -> None:
    passages = [
        "Apples and pears grow in the orchard.",
        "The lunar surface contains many impact craters.",
        "Carrots are common root vegetables.",
    ]

    ranking = rank_bm25("LUNAR craters", passages)

    assert ranking[0][0] == 1
    assert ranking[0][1] > ranking[1][1]


def test_rank_with_bm25_breaks_ties_by_original_order() -> None:
    passages = ["alpha one", "beta two", "gamma three"]

    rank1 = rank_bm25("unmatched", passages)
    rank2 = rank_bm25("unmatched", passages)

    assert [idx for idx, _ in rank1] == [0, 1, 2]
    assert rank1 == rank2


def test_rank_with_bm25_rejects_empty_passage_list() -> None:
    with pytest.raises(ValueError, match="passages must not be empty"):
        rank_bm25("query", [])


def test_rank_with_bm25_rejects_passages_without_tokens() -> None:
    with pytest.raises(
        ValueError, match="passages must contain at least one BM25 token"
    ):
        rank_bm25("query", ["...", "---"])


class FakeE5Tokenizer:
    def __call__(
        self, text: str, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[int]]:
        assert add_special_tokens is True
        assert truncation is False

        token_count = len(text.split()) + 2
        return {"input_ids": list(range(token_count))}


def test_format_e5_query_adds_required_prefix() -> None:
    assert fmt_e5_query("How do I apply?") == "query: How do I apply?"


def test_format_e5_passage_adds_required_prefix() -> None:
    assert (
        fmt_e5_passage("Applications are submitted online.")
        == "passage: Applications are submitted online."
    )


def test_validate_e5_length_accepts_exact_limit() -> None:
    tokenizer = FakeE5Tokenizer()
    passage = " ".join(["word"] * 509)
    fmt_passage = fmt_e5_passage(passage)

    assert check_e5_length(fmt_passage, tokenizer) == E5_MAX_TOKENS


def test_validate_e5_length_rejects_over_limit() -> None:
    tokenizer = FakeE5Tokenizer()
    passage = " ".join(["word"] * 510)
    fmt_passage = fmt_e5_passage(passage)

    with pytest.raises(ValueError, match="E5 input has 513 tokens"):
        check_e5_length(fmt_passage, tokenizer)


class FakeE5:
    def __init__(self, embeddings) -> None:
        self.tokenizer = FakeE5Tokenizer()
        self.embeddings = np.asarray(embeddings, dtype=float)
        self.encoded_texts = None

    def encode(
        self,
        inputs,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ):
        assert normalize_embeddings is True
        assert convert_to_numpy is True
        assert show_progress_bar is False

        self.encoded_texts = list(inputs)
        return self.embeddings


def test_load_e5_model_uses_pinned_revision_and_cpu(monkeypatch) -> None:
    calls = {}

    def fake_sentence_transformer(model_name: str, revision: str, device: str):
        calls["model_name"] = model_name
        calls["revision"] = revision
        calls["device"] = device
        return object()

    monkeypatch.setattr(retrieval, "SentenceTransformer", fake_sentence_transformer)

    model = retrieval.load_e5()

    assert model is not None
    assert calls == {
        "model_name": retrieval.E5_MODEL_NAME,
        "revision": retrieval.E5_MODEL_REVISION,
        "device": "cpu",
    }


def test_rank_with_e5_formats_inputs_and_ranks_by_similarity() -> None:
    model = FakeE5([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
    passages = ["first passage", "best passage", "second passage"]

    ranking = rank_e5("test query", passages, model)

    assert [idx for idx, _ in ranking] == [1, 2, 0]
    assert model.encoded_texts == [
        "query: test query",
        "passage: first passage",
        "passage: best passage",
        "passage: second passage",
    ]


def test_rank_with_e5_breaks_ties_by_original_order() -> None:
    model = FakeE5([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    ranking = rank_e5("test query", ["first", "second", "third"], model)

    assert [idx for idx, _ in ranking] == [0, 1, 2]


def test_rank_with_e5_rejects_empty_passage_list() -> None:
    model = FakeE5([[1.0, 0.0]])

    with pytest.raises(ValueError, match="passages must not be empty"):
        rank_e5("query", [], model)


def test_rank_with_e5_checks_length_before_encoding() -> None:
    model = FakeE5([[1.0, 0.0], [1.0, 0.0]])
    long_passage = " ".join(["word"] * 510)

    with pytest.raises(ValueError, match="E5 input has 513 tokens"):
        rank_e5("query", [long_passage], model)

    assert model.encoded_texts is None
