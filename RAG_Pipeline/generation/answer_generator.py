import os
import joblib
from typing import List

from RAG_Pipeline.rag_utils import generate_context, generate_answer
from utils import get_ground_truths

class RAGPredictor:
    def __init__(self, chunk_data, embedder, index, metadata_list, llm):
        self.chunk_data = chunk_data
        self.embedder = embedder
        self.index = index
        self.metadata_list = metadata_list
        self.llm = llm

    def run(self, start: int, end: int, k: int = 1, save_dir: str = "./save_test/", save_prefix: str = "predictions", save_interval: int = 100):
        os.makedirs(save_dir, exist_ok=True)

        all_predictions = []
        predictions = []

        for i in range(start, end):
            q = self.chunk_data[i]["question"]
            context = generate_context(self.embedder, q, self.index, self.metadata_list, k=k, mode="context")
            response = generate_answer(self.llm, context, q)
            predictions.append(response)

            print(f"[Progress] Question {i + 1} / {end} completed.")

            if (i + 1) % save_interval == 0:
                part_filename = f"{save_prefix}_{i + 1 - save_interval}_{i + 1}.joblib"
                joblib.dump(predictions, os.path.join(save_dir, part_filename), compress=0)
                all_predictions.extend(predictions)
                predictions.clear()

        # save remaining
        all_predictions.extend(predictions)
        full_filename = f"{save_prefix}_full_results_{start}_{end}.joblib"
        joblib.dump(all_predictions, os.path.join(save_dir, full_filename), compress=0)

        print("[Done] All predictions saved. Total:", len(all_predictions))
        return all_predictions