import re

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

E5_MODEL_NAME = "intfloat/e5-base-v2"
E5_MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
E5_MAX_TOKENS = 512


def bm25_tokens(text: str) -> list[str]:
    """Tokenize text for BM25 using lowercase regex word tokens."""
    return re.findall(r"\w+", text.lower())


def rank_bm25(query: str, passages: list[str]) -> list[tuple[int, float]]:
    """Rank passages for a query with Okapi BM25."""
    if not passages:
        raise ValueError("passages must not be empty")

    tokenized_passages = [bm25_tokens(passage) for passage in passages]

    if not any(tokenized_passages):
        raise ValueError("passages must contain at least one BM25 token")

    bm25 = BM25Okapi(tokenized_passages, k1=BM25_K1, b=BM25_B, epsilon=BM25_EPSILON)
    scores = bm25.get_scores(bm25_tokens(query))

    ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))

    return [(passage_idx, float(score)) for passage_idx, score in ranked]


def fmt_e5_query(query: str) -> str:
    """Add the E5 retrieval prefix to a query."""
    return f"query: {query}"


def fmt_e5_passage(passage: str) -> str:
    """Add the E5 retrieval prefix to a passage."""
    return f"passage: {passage}"


def check_e5_length(text: str, tokenizer) -> int:
    """Count E5 input tokens and reject inputs over the limit."""
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    token_count = len(encoded["input_ids"])

    if token_count > E5_MAX_TOKENS:
        raise ValueError(
            f"E5 input has {token_count} tokens; maximum is {E5_MAX_TOKENS}"
        )

    return token_count


def load_e5() -> SentenceTransformer:
    """Load the fixed E5 model revision on CPU."""
    return SentenceTransformer(E5_MODEL_NAME, revision=E5_MODEL_REVISION, device="cpu")


def rank_e5(
    query: str, passages: list[str], model: SentenceTransformer
) -> list[tuple[int, float]]:
    """Rank passages using normalized E5 embeddings."""
    if not passages:
        raise ValueError("passages must not be empty")

    fmt_inputs = [fmt_e5_query(query)]
    fmt_inputs.extend(fmt_e5_passage(passage) for passage in passages)

    for text in fmt_inputs:
        check_e5_length(text, model.tokenizer)

    embeddings = model.encode(
        fmt_inputs,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = embeddings[0]
    passage_embeddings = embeddings[1:]
    scores = passage_embeddings @ query_embedding

    ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))

    return [(passage_idx, float(score)) for passage_idx, score in ranked]
