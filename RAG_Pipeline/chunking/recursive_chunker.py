from langchain_text_splitters import RecursiveCharacterTextSplitter

def word_count(text):
    return len(text.split())

def chunk_text_recursive(
        text: str,
        chunk_size: int = 100,
        overlap: int = 30):

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[". ", "\n\n", "\n", ".", "?", "!", ";"],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=word_count,
        keep_separator=True
    )

    chunks = text_splitter.split_text(text)

    return chunks