from collections import Counter
import re

class Evaluator:
    def __init__(self, pred, answers):
        self.pred = pred
        self.answers = answers

    def exact_match(self):
        """
        Returns 1 if the prediction fully or partially matches any of the ground truth answers.
        """
        pred = self.pred.strip().lower()
        # If the predicted answer contains any ground truth answer, count it as correct
        return int(any(ref in pred for ref in self.answers))


    def f1_score(self):
        """
        Computes the F1 score by finding the best token overlap match.
        First extracts the most relevant part of the prediction.
        """
        pred = self.extract_full_best_answer()  # Extract best answer from prediction
        pred_tokens = pred.lower().split()

        best_f1 = 0
        for ref in self.answers:
            if isinstance(ref, list):
                ref = " ".join(ref)
            ref_tokens = ref.lower().split()

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(ref_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
            best_f1 = max(best_f1, f1)

        return best_f1

    def extract_full_best_answer(self):
        """
        Extracts the most relevant substring from the prediction.
        Ensures that multi-word entities (e.g., "Sinclair Lewis") are correctly extracted.
        """
        pred = self.pred.lower()

        # If a ground truth answer appears in the prediction, return the best match
        for ref in self.answers:
            if ref in pred:
                return ref  # Directly return the best-matching entity

        # Otherwise, try to extract a full proper noun (capitalized words)
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', pred)

        # Return the longest match (most likely to be a full name)
        return max(matches, key=len) if matches else pred