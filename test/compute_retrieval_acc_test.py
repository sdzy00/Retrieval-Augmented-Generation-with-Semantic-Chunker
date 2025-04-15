import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

from datasets import load_from_disk

from evaluation.compute_retrieval_acc import test_compute_retrieval_accuracy, retrieve_docs_list


class MyTestCase(unittest.TestCase):
    def test_compute_retrieval_acc(self):
        # metadata_path_fixed_size = "../data/metadata/metadata_list_fixed_size_bge-large-en-v1.5_chunksize200_overlap20_random500data.joblib"
        # embedding_path_fixed_size = "../data/embeddings/embeddings_fixed_size_bge-large-en-v1.5_chunksize200_overlap20_test500random.npy"
        #
        # metadata_path_recursive = "../data/metadata/metadata_list_recursive_bge-large-en-v1.5_chunksize200_overlap20_random500data.joblib"
        # embedding_path_recursive = "../data/embeddings/embeddings_recursive_bge-large-en-v1.5_chunksize200_overlap20_test500random.npy"

        metadata_path_semantic = "data/metadata/metadata_list_semantic_bge-large-en-v1.5_random500data.joblib"
        embedding_path_semantic = "data/embeddings/embeddings_semantic_bge-large-en-v1.5_test500random.npy"

        data_path = "data/chunk_random500"
        data = load_from_disk(data_path)

        # embedder_fixed_size, index_fixed_size, metadata_list_fixed_size = retrieve_docs_list(metadata_path_fixed_size,
        #                                                                                      embedding_path_fixed_size)
        # test_compute_retrieval_accuracy(data, embedder_fixed_size, index_fixed_size, metadata_list_fixed_size,
        #                                 method="fixed_size", length=len(data))
        #
        # embedder_recursive, index_recursive, metadata_list_recursive = retrieve_docs_list(metadata_path_recursive,
        #                                                                                   embedding_path_recursive)
        # test_compute_retrieval_accuracy(data, embedder_recursive, index_recursive, metadata_list_recursive,
        #                                 method="recursive", length=len(data))

        embedder_semantic, index_semantic, metadata_list_semantic = retrieve_docs_list(metadata_path_semantic,
                                                                                       embedding_path_semantic)
        test_compute_retrieval_accuracy(data, embedder_semantic, index_semantic, metadata_list_semantic,
                                        method="semantic", length=len(data))


if __name__ == '__main__':
    unittest.main()
