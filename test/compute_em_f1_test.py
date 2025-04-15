import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

import joblib
from datasets import load_from_disk

from evaluation.compute_em_f1 import Evaluate


class MyTestCase(unittest.TestCase):
    def test_compute_em_f1(self):
        data_path = "data/chunk_random500"
        data = load_from_disk(data_path)

        # response_fixed = "../data/predictions/fixed_size_chunking/chunksize200_overlap20_random500data/predictions_all-MiniLM-L6-v2_fixed_size_chunking_maxlen100_k=1_full_results_0_500.joblib"
        # response_recursive = "../data/predictions/recursive_chunking/chunksize200_overlap20_random500data/predictions_all-MiniLM-L6-v2_recursive_chunking_maxlen100_k=1_full_results_0_500.joblib"
        response_semantic = "data/predictions/semantic_chunking/recursive_chunksize3000_overlap200_random500data/predictions_all-MiniLM-L6-v2_semantic_chunking_maxlen100_k=1_full_results_0_500.joblib"

        # evaluate_fixed_size = Evaluate(data, response_fixed)
        # evaluate_fixed_size.compute_em_f1()
        #
        # evaluate_recursive = Evaluate(data, response_recursive)
        # evaluate_recursive.compute_em_f1()

        evaluate_semantic = Evaluate(data, response_semantic)
        evaluate_semantic.compute_em_f1()

if __name__ == '__main__':
    unittest.main()
