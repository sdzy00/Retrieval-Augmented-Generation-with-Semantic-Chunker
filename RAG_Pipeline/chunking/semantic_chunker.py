from langchain_experimental.text_splitter import SemanticChunker

def chunk_text_semantic(text, embedder):
    text_splitter = SemanticChunker(embeddings=embedder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=85.0, min_chunk_size=1000)
    chunks = text_splitter.create_documents([text])
    return chunks



