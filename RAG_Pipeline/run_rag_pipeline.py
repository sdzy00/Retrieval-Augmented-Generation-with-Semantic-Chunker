import copy
import torch
import joblib
from datasets import load_from_disk
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

from RAG_Pipeline.chunking.Chunker import chunk_data
from RAG_Pipeline.generation.answer_generator import RAGPredictor
from RAG_Pipeline.rag_utils import extract_text_from_chunk, clean_text, generate_metadata_list, store_embeddings
from RAG_Pipeline.retrieval.indexing import EmbeddingIndexer

class RAG_Pipeline:
    def __init__(self, chunk_path, metadata_path, embedding_path, save_path):
        self.chunk_path = chunk_path
        self.metadata_path = metadata_path
        self.embedding_path = embedding_path
        self.save_path = save_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def start(self, method):
        chunk = load_from_disk(self.chunk_path)
        print(f"Loaded {len(chunk)} samples")

        # Step 1: Clean and Chunk
        docs = extract_text_from_chunk(chunk)
        cleaned_docs = copy.deepcopy(docs)
        for doc in cleaned_docs:
            # print(type(doc["text"]))
            doc["text"] = [clean_text(text) for text in doc["text"] if text]

        chunked_docs = chunk_data(cleaned_docs, method=method)
        # joblib.dump(chunked_docs, f"../{datapath}/predictions/save_test/chunk_semantic_first3data.joblib", compress=0)

        # Step 2: Generate Metadata
        generate_metadata_list(chunked_docs, self.metadata_path)

        # Step 3: Generate Embeddings
        model_name = 'BAAI/bge-large-en-v1.5'
        embedder = SentenceTransformer(model_name, device=self.device)

        embeddings = store_embeddings(chunked_docs,
            embeddings_path=self.embedding_path,
            embedder=embedder)

        # Step 4: Build Index
        embedding_indexer = EmbeddingIndexer(model_name, self.embedding_path, self.metadata_path)
        embedder, index, metadata_list = embedding_indexer.get()

        # Step 5: Generate Answers
        chat = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=100,
            api_key=""
        )
        rag_generator = RAGPredictor(chunk, embedder, index, metadata_list, chat)
        rag_generator.run(start=0, end=3, k=1, save_dir=self.save_path,
                          save_prefix="predictions_bge-large-en-v1.5_semantic_chunking_minlen1000chars_k=1")

