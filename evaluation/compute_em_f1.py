import joblib

from evaluation.evaluator import Evaluator
from utils import normalize_decade_answer, get_ground_truths

class Evaluate:
    def __init__(self, data, prediction_path:str):
        self.data = data
        self.prediction_path = prediction_path


    def compute_em_f1(self):
        response = joblib.load(self.prediction_path)
        print("response length:", len(response))

        references_list = []
        wrong_pred_list = []
        predictions_list = []

        for i in range(0, len(response)):
            answer = get_ground_truths(self.data[i]["answer"])
            evaluator = Evaluator(response[i].content, answer)
            filtered_prediction = evaluator.extract_full_best_answer()
            normalized_pred = normalize_decade_answer(filtered_prediction, answer)

            predictions_list.append(normalized_pred)
            references_list.append(answer)

            instance_evaluator = Evaluator(normalized_pred, answer)
            if instance_evaluator.exact_match() < 1.0 or instance_evaluator.f1_score() < 1.0:
                wrong_pred_list.append({"wrong_pred": normalized_pred, "ground_truth": answer})


        em_scores = [Evaluator(pred, ref).exact_match() for pred, ref in zip(predictions_list, references_list)]
        print(f"Exact Match (EM): {sum(em_scores) / len(em_scores):.4f}")
        f1_scores = [Evaluator(pred, ref).f1_score() for pred, ref in zip(predictions_list, references_list)]
        print(f"F1 Score: {sum(f1_scores) / len(f1_scores):.4f}")


