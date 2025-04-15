import faiss
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

from RAG_Pipeline.rag_utils import load_embeddings_local, load_metadata_list

class EmbeddingIndexer:
    def __init__(self, model_name: str, embedding_path: str, metadata_path: str):
        self.model_name = model_name
        self.embedding_path = embedding_path
        self.metadata_path = metadata_path

        self.embedder = None
        self.index = None
        self.metadata_list = None

        self._build()

    def _build(self):
        print(f"[Indexer] Loading metadata from: {self.metadata_path}")
        self.metadata_list = load_metadata_list(self.metadata_path)

        print(f"[Indexer] Loading embedder: {self.model_name}")
        self.embedder = SentenceTransformer(self.model_name)

        print(f"[Indexer] Loading embeddings from: {self.embedding_path}")
        embeddings = load_embeddings_local(self.embedding_path)
        print(f"[Indexer] Embedding shape: {embeddings.shape}")

        embedding_dim = embeddings.shape[1]
        faiss_embeddings = embeddings.astype(np.float32)

        base_index = faiss.IndexFlatL2(embedding_dim)

        index = faiss.IndexIDMap(base_index)
        ids = np.arange(len(faiss_embeddings)).astype(np.int64)
        index.add_with_ids(faiss_embeddings, ids)

        print(f"[Indexer] Added {index.ntotal} vectors to FAISS index.")
        self.index = index

    def get(self):
        return self.embedder, self.index, self.metadata_list
