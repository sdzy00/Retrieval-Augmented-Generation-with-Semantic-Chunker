import re

import joblib

def is_decade(answer):
    patterns = [
        r'\b\d{3}0s\b',           # "1930s"
        r'\b\d{2}0s\b',           # "30s"
        r'\b\d{2}\s?0s\b',        # "30 s"
        r'\b\d{2}\s\d{2}\b',      # "30 39"
        r'\b\d{2}0s\s\w+',        # "30s ad"
    ]
    return any(re.match(p, answer.strip().lower()) for p in patterns)

def year_to_decade(year):
    year = int(year)
    decade = (year // 10) * 10
    return f"{decade}s"

def normalize_decade_answer(pred, ground_truths):
    contains_decade = any(is_decade(gt) for gt in ground_truths)

    if contains_decade:
        years_in_pred = re.findall(r'\b(?:18|19|20)\d{2}\b', pred)
        if years_in_pred := re.findall(r'\b(?:18|19|20)\d{2}\b', pred):
            pred_decade = year_to_decade(years_in_pred[0])
            return pred_decade

    return pred

def get_ground_truths(answer):
    return [answer["normalized_value"]] + answer["normalized_aliases"]

