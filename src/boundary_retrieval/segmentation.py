import re

from document_chunking_utility.chunkers import TextChunk, chunk_by_fixed_words
from document_chunking_utility.validation import validate_chunk_settings


def chunk_shifted(
    text: str, chunk_size: int, phase: int, overlap: int = 0
) -> list[TextChunk]:
    """Split text into fixed-word chunks with boundaries shifted by phase words."""
    validate_chunk_settings(chunk_size=chunk_size, overlap=overlap)

    step = chunk_size - overlap

    if phase < 0 or phase >= step:
        raise ValueError("phase must be between 0 and chunk stride - 1")

    if phase == 0:
        return chunk_by_fixed_words(text, chunk_size=chunk_size, overlap=overlap)

    words = list(re.finditer(r"\S+", text))

    # Keep a short document as one chunk when it ends before the shift.
    if phase >= len(words):
        return chunk_by_fixed_words(text, chunk_size=chunk_size, overlap=overlap)

    prefix_start = words[0].start()
    prefix_word_count = min(phase + overlap, len(words))
    prefix_end = words[prefix_word_count - 1].end()

    prefix_chunk = TextChunk(
        chunk_index=0,
        chunk_text=text[prefix_start:prefix_end],
        start_char=prefix_start,
        end_char=prefix_end,
    )

    suffix_start = words[phase].start()
    suffix_text = text[suffix_start:]

    suffix_chunks = chunk_by_fixed_words(
        suffix_text, chunk_size=chunk_size, overlap=overlap
    )

    shifted_chunks = [prefix_chunk]

    # Map suffix offsets back to the original document.
    for chunk in suffix_chunks:
        shifted_chunks.append(
            TextChunk(
                chunk_index=len(shifted_chunks),
                chunk_text=chunk.chunk_text,
                start_char=suffix_start + chunk.start_char,
                end_char=suffix_start + chunk.end_char,
            )
        )

    return shifted_chunks
