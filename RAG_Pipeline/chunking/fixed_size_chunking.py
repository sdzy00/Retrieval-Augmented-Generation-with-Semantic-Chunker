def chunk_text_by_words_fixed(
    text: str,
    chunk_size: int = 100,
    overlap: int = 30
):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()

        if chunk_str:
            chunks.append(chunk_str)

        start += (chunk_size - overlap) if (chunk_size - overlap) > 0 else chunk_size

    return chunks