import unittest

from RAG_Pipeline.run_rag_pipeline import RAG_Pipeline


class MyTestCase(unittest.TestCase):
    def test_rag_pipeline(self):
        datapath = "../data"
        chunk_path = datapath + "/chunk_first3"
        metadata_path = datapath + "/metadata/metadata_list_semantic_bge-large-en-v1.5_first3data.joblib"
        embedding_path = datapath + "/embeddings/embeddings_semantic_bge-large-en-v1.5_first3data.npy"
        save_path = datapath + "/predictions/save_test"
        rag_pipeline = RAG_Pipeline(chunk_path, metadata_path, embedding_path, save_path)
        rag_pipeline.start(method="semantic_chunk") # method: "semantic_chunk", "fixed_size", "recursive_chunk"

if __name__ == '__main__':
    unittest.main()
