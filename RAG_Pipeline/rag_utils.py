import joblib
import numpy as np
import html
import re

def generate_answer(chat, context, query):
    prompt_text = f"Context: {context}\n\nQuestion: {query}"
    return chat.invoke(prompt_text)

def generate_context(embedder, query, index, metadata_list, mode="context", k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    # assert query_embedding.shape[1] == index.d, \
    #     f"Dimension mismatch: query dimension {query_embedding.shape[1]} vs index dimension {index.d}"

    # k = 3  # number of hits
    D, I = index.search(query_embedding, k)  # D = distances, I = indices

    # print("Top K indices:", I[0])
    # print("Distances:", D[0])

    context = "\n\n"
    # Get the corresponding metadata
    retrieved_docs = []
    for idx in I[0]:
        data_retrieved = metadata_list[idx]
        # print(data_retrieved)  # This chunk doc is relevant
        if mode == "retrieved_docs":
            retrieved_docs.append(data_retrieved["text"])
        else:
            retrieved_docs.append(data_retrieved)

    if mode == "retrieved_docs":
        return retrieved_docs

    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    return context

def load_metadata_list(datapath):
    return joblib.load(datapath)

def load_embeddings_local(embeddings_file):
    return np.load(embeddings_file)

def store_embeddings(chunked_docs, embeddings_path, embedder, batch_size=32):
    embeddings = []

    for i in range(0, len(chunked_docs), batch_size):
        batch_docs = chunked_docs[i: i + batch_size]
        batch_texts = [d["text"] for d in batch_docs]

        # Encode returns a numpy array of shape (batch_size, embedding_dim)
        batch_embeddings = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        print(f"Processed {i // 32} / {len(chunked_docs) // batch_size + 1} documents")

    # Concatenate all embeddings into one array
    embeddings = np.concatenate(embeddings, axis=0)  # shape: (num_chunks, embedding_dim)
    # Save embeddings
    np.save(embeddings_path, embeddings)
    return embeddings

def generate_metadata_list(chunked_docs, save_path, batch_size=32):
    metadata_list = []

    for i in range(0, len(chunked_docs), batch_size):
        batch_docs = chunked_docs[i: i + batch_size]
        # Encode returns a numpy array of shape (batch_size, embedding_dim)
        metadata_list.extend(batch_docs)
        # print(f"Processed {i // 32} / {len(chunked_docs) // batch_size + 1} documents")

    joblib.dump(metadata_list, save_path, compress=0)
    print(f"Metadata list saved to {save_path}")

    return metadata_list

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_chunk(chunk):
    docs = []

    for example in chunk:
        question_id = example["question_id"]

        # Wikipedia source
        wiki_data = example.get("entity_pages", {})
        if wiki_data:
            docs.append({
                "question_id": question_id,
                "source": "wikipedia",
                "title": wiki_data.get("title", ""),
                "url": None,
                "text": wiki_data.get("wiki_context", [])
            })

        # Web source
        web_data = example.get("search_results", {})
        if web_data:
            docs.append({
                "question_id": question_id,
                "source": "web",
                "title": web_data.get("title", ""),
                "url": web_data.get("url", None),
                "text": web_data.get("search_context", [])
            })

    print(f"Extracted {len(docs)} docs from chunk.")
    return docs

def clean_docs(doc_list):
    cleaned_docs = []
    for doc in doc_list:
        cleaned = doc.copy()
        cleaned["text"] = [clean_text(t) for t in doc["text"] if t]
        cleaned_docs.append(cleaned)
    return cleaned_docs

def clean_chunk_punctuation(chunks):
    cleaned_chunks = []
    for chunk in chunks:
        chunk = chunk.lstrip(". ")
        if not chunk.endswith((".", "!", "?")):
            chunk += "."
        cleaned_chunks.append(chunk)

    return cleaned_chunks