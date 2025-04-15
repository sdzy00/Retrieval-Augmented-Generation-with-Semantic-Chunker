from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import torch

from RAG_Pipeline.chunking.fixed_size_chunking import chunk_text_by_words_fixed
from RAG_Pipeline.chunking.recursive_chunker import chunk_text_recursive
from RAG_Pipeline.chunking.semantic_chunker import chunk_text_semantic
from RAG_Pipeline.rag_utils import clean_chunk_punctuation


def chunk_data(cleaned_docs, method="fixed_size"):
    print("chunk method: ", method)
    chunked_docs = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    for j, doc in enumerate(cleaned_docs):
        # text = ""
        # for sentence in doc["text"]:
        #     text += " " + sentence
        text = " ".join(doc["text"])

        if method == "semantic_chunk":
            intermediate_chunks = chunk_text_recursive(text, chunk_size=3000, overlap=200)
            chunks = []
            for chunk in intermediate_chunks:
                chunks.extend(chunk_text_semantic(chunk, hf))
            # chunks = chunk_text_semantic(text)
            for i, chunk_text in enumerate(chunks):
                chunk_doc = {
                    "question_id": doc["question_id"],
                    "source": doc["source"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "chunk_index": i,
                    "text": chunk_text.page_content
                }
                chunked_docs.append(chunk_doc)
            print("Processed: {} / {}".format(j + 1, len(cleaned_docs)))
            # if j + 1 == 500:
            #   chunk_path = datapath + "/chunk_semantic_random500data_first_half.joblib"
            #   joblib.dump(chunked_docs, chunk_path, compress=0)
            continue

        elif method == "fixed_size":
            chunks = chunk_text_by_words_fixed(text, chunk_size=500, overlap=50)
        else:
            chunks = chunk_text_recursive(text, chunk_size=500, overlap=50)
            chunks = clean_chunk_punctuation(chunks)

        # For each chunk, create a new dict so we can keep metadata
        for i, chunk_text in enumerate(chunks):
            chunk_doc = {
                "question_id": doc["question_id"],
                "source": doc["source"],
                "title": doc["title"],
                "url": doc["url"],
                "chunk_index": i,
                "text": chunk_text
            }
            chunked_docs.append(chunk_doc)

    print(f"Total chunked docs: {len(chunked_docs)}")
    return chunked_docs
