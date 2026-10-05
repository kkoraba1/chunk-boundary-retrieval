import re

import pytest
from document_chunking_utility.chunkers import TextChunk, chunk_by_fixed_words

from boundary_retrieval.segmentation import chunk_shifted


def test_phase_zero_matches_fixed_word_chunker() -> None:
    text = "alpha  beta\ngamma\tdelta epsilon zeta eta theta iota kappa"

    expected = chunk_by_fixed_words(text, chunk_size=4, overlap=0)
    got = chunk_shifted(text, chunk_size=4, phase=0, overlap=0)

    assert got == expected


def test_phase_zero_with_overlap_matches_fixed_word_chunker() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"

    expected = chunk_by_fixed_words(text, chunk_size=4, overlap=1)
    got = chunk_shifted(text, chunk_size=4, phase=0, overlap=1)

    assert got == expected


def test_nonzero_phase_keeps_prefix_and_remaps_offsets() -> None:
    text = "zero  one\ntwo three\tfour five six seven"

    chunks = chunk_shifted(text, chunk_size=3, phase=2, overlap=0)

    assert chunks == [
        TextChunk(chunk_index=0, chunk_text="zero  one", start_char=0, end_char=9),
        TextChunk(
            chunk_index=1, chunk_text="two three\tfour", start_char=10, end_char=24
        ),
        TextChunk(
            chunk_index=2, chunk_text="five six seven", start_char=25, end_char=39
        ),
    ]


def test_nonzero_phase_with_overlap_uses_shifted_stride() -> None:
    words = [f"word{i}" for i in range(12)]
    text = " ".join(words)

    chunks = chunk_shifted(text, chunk_size=5, phase=2, overlap=2)

    got_groups = [re.findall(r"\S+", chunk.chunk_text) for chunk in chunks]

    assert got_groups == [words[0:4], words[2:7], words[5:10], words[8:12]]


def test_all_source_words_are_kept_across_phases() -> None:
    text = (
        "alpha  bravo\ncharlie delta\techo foxtrot golf hotel "
        "india juliet kilo lima mike"
    )
    src_words = re.findall(r"\S+", text)

    for phase in range(5):
        chunks = chunk_shifted(text, chunk_size=5, phase=phase, overlap=0)

        chunk_words = [
            word for chunk in chunks for word in re.findall(r"\S+", chunk.chunk_text)
        ]

        assert chunk_words == src_words


def test_all_source_words_are_covered_with_overlap() -> None:
    words = [f"word{i}" for i in range(18)]
    text = " ".join(words)

    for phase in range(4):
        chunks = chunk_shifted(text, chunk_size=6, phase=phase, overlap=2)

        covered = {
            word for chunk in chunks for word in re.findall(r"\S+", chunk.chunk_text)
        }

        assert covered == set(words)


def test_chunk_offsets_match_original_document_across_phases() -> None:
    text = (
        "  alpha  bravo\ncharlie delta\techo foxtrot golf hotel "
        "india juliet kilo lima mike  "
    )

    for phase in range(5):
        chunks = chunk_shifted(text, chunk_size=5, phase=phase, overlap=0)

        for chunk in chunks:
            assert text[chunk.start_char : chunk.end_char] == chunk.chunk_text


def test_chunk_offsets_match_original_document_with_overlap() -> None:
    text = (
        "  alpha  bravo\ncharlie delta\techo foxtrot golf hotel "
        "india juliet kilo lima mike  "
    )

    for phase in range(4):
        chunks = chunk_shifted(text, chunk_size=5, phase=phase, overlap=1)

        for chunk in chunks:
            assert text[chunk.start_char : chunk.end_char] == chunk.chunk_text


def test_invalid_phases_raise_value_error() -> None:
    text = "alpha beta gamma delta epsilon"

    with pytest.raises(
        ValueError, match="phase must be between 0 and chunk stride - 1"
    ):
        chunk_shifted(text, chunk_size=5, phase=-1, overlap=0)

    with pytest.raises(
        ValueError, match="phase must be between 0 and chunk stride - 1"
    ):
        chunk_shifted(text, chunk_size=5, phase=5, overlap=0)


def test_overlap_phase_must_fit_chunk_stride() -> None:
    text = "alpha beta gamma delta epsilon zeta"

    with pytest.raises(
        ValueError, match="phase must be between 0 and chunk stride - 1"
    ):
        chunk_shifted(text, chunk_size=5, phase=3, overlap=2)


def test_whitespace_only_text_returns_no_chunks() -> None:
    text = "  \n\t   "

    for phase in range(4):
        chunks = chunk_shifted(text, chunk_size=4, phase=phase, overlap=0)

        assert chunks == []


def test_phase_at_document_end_keeps_short_document_whole() -> None:
    text = "alpha beta gamma"

    expected = chunk_by_fixed_words(text, chunk_size=5, overlap=0)

    for phase in (3, 4):
        got = chunk_shifted(text, chunk_size=5, phase=phase, overlap=0)

        assert got == expected


def test_primary_experiment_phases_place_boundaries_correctly() -> None:
    words = [f"word{i}" for i in range(430)]
    text = " ".join(words)

    chunk_size = 200
    phases = (0, 25, 50, 75, 100, 125, 150, 175)

    for phase in phases:
        chunks = chunk_shifted(text, chunk_size=chunk_size, phase=phase, overlap=0)

        expected_groups = []

        if phase > 0:
            expected_groups.append(words[:phase])

        for start in range(phase, len(words), chunk_size):
            expected_groups.append(words[start : start + chunk_size])

        got_groups = [re.findall(r"\S+", chunk.chunk_text) for chunk in chunks]

        assert got_groups == expected_groups


def test_overlap_ablation_phases_cover_one_stride_cycle() -> None:
    words = [f"word{i}" for i in range(520)]
    text = " ".join(words)

    chunk_size = 200
    overlap = 40
    phases = (0, 40, 80, 120)
    stride = chunk_size - overlap

    for phase in phases:
        chunks = chunk_shifted(
            text, chunk_size=chunk_size, phase=phase, overlap=overlap
        )

        expected_groups = []

        if phase > 0:
            prefix_end = min(phase + overlap, len(words))
            expected_groups.append(words[:prefix_end])

        start = phase

        while start < len(words):
            expected_groups.append(words[start : start + chunk_size])

            if start + chunk_size >= len(words):
                break

            start += stride

        got_groups = [re.findall(r"\S+", chunk.chunk_text) for chunk in chunks]

        assert got_groups == expected_groups
