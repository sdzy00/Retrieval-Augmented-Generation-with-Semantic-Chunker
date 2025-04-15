import re
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from RAG_Pipeline.rag_utils import generate_context, load_metadata_list, load_embeddings_local


def get_ground_truths(answer):
    """
    Extracts normalized ground truth answers from the TriviaQA answer format.
    """
    return [answer["normalized_value"]] + answer["normalized_aliases"]

def normalize_text(text):
    """
    Normalize text by lowercasing, stripping, and removing punctuation.
    """
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

def contains_answer(chunk, answer):
    """
    Check if the normalized answer is contained within the normalized chunk.
    """
    norm_chunk = normalize_text(chunk)
    norm_answer = normalize_text(answer)
    return norm_answer in norm_chunk

def compute_retrieval_accuracy(queries, ground_truths, retrieved_chunks_list):
    """
    Compute retrieval accuracy as the proportion of queries where at least one retrieved chunk
    contains the correct answer.

    Parameters:
        queries: list of query strings.
        ground_truths: list of ground truth answers (each can be a string or a list of acceptable strings).
        retrieved_chunks_list: list where each element is a list of retrieved chunks (strings) for the corresponding query.

    Returns:
        retrieval_accuracy: float, the proportion of queries with correct retrieval.
    """
    total = len(queries)
    count_with_answer = 0

    for gt, chunks in zip(ground_truths, retrieved_chunks_list):
        # If ground truth is provided as a list of acceptable answers
        if isinstance(gt, list):
            found = any(any(contains_answer(chunk, a) for chunk in chunks) for a in gt)
        else:
            found = any(contains_answer(chunk, gt) for chunk in chunks)

        if found:
            count_with_answer += 1
        # else:
        #     print("ground truth:", gt)
        #     print("retrieved chunks:", chunks)

    # print(f"Retrieved {count_with_answer} / {total} answers")
    retrieval_accuracy = count_with_answer / total
    return retrieval_accuracy

def retrieve_docs_list(metadata_path_to_retrieve, embedding_path_to_retrieve):
    metadata_list_to_retrieve = load_metadata_list(metadata_path_to_retrieve)

    # model_name_to_retrieve = "sentence-transformers/all-MiniLM-L6-v2"
    model_name_to_retrieve = "BAAI/bge-large-en-v1.5"
    embedder_to_retrieve = SentenceTransformer(model_name_to_retrieve)

    embeddings_to_retrieve = load_embeddings_local(embedding_path_to_retrieve)

    embedding_dim_to_retrieve = embeddings_to_retrieve.shape[1]

    # Initialize a FAISS index with L2 distance
    index_to_retrieve = faiss.IndexFlatL2(embedding_dim_to_retrieve)

    # Convert embeddings to float32 for FAISS
    faiss_embeddings_to_retrieve = embeddings_to_retrieve.astype(np.float32)

    # Add embeddings to index
    index_to_retrieve.add(faiss_embeddings_to_retrieve)

    return embedder_to_retrieve, index_to_retrieve, metadata_list_to_retrieve

def test_compute_retrieval_accuracy(chunk_local, embedder_local, index_local, metadata_list_local, method="fixed_size", length=2000):
    queries = []
    ground_truths = []
    # Example retrieved chunks for the above query
    retrieved_chunks_list = []

    start, end = 0, length
    for i in range(start, end):
        q = chunk_local[i]["question"]
        answer = get_ground_truths(chunk_local[i]["answer"])
        context = generate_context(embedder_local, q, index_local, metadata_list_local, k=1, mode="retrieved_docs")
        queries.append(q)
        ground_truths.append(answer)
        retrieved_chunks_list.append(context)
        # print("query:", q)
        # print("answer:", answer)
        # print("context:", context)

    accuracy = compute_retrieval_accuracy(queries, ground_truths, retrieved_chunks_list)
    print(f"Retrieval Accuracy_{method} : {accuracy:.4f}")