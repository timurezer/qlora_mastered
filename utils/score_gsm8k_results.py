"""Score extracted GSM8K results (results_*.json from extract_evalscope_outputs_gsm8k.py).

Same comparison as qlora_clear's accuracy_eval/get_accuracy.py --nums_compare, but rows
whose extracted answer is not a number count as wrong instead of crashing the whole run
(the pipeline's get_accuracy.py dies on outputs like '0.1012012.00').

Usage: python score_gsm8k_results.py <results.json>
"""
import json
import re
import sys


def main():
    path = sys.argv[1]
    dataset = json.load(open(path, encoding="utf-8"))
    correct = 0
    unparsable = 0
    for row in dataset:
        answer = re.sub(r"[^\d.-]", "", row["output"].replace("【", "").replace("】", "").strip())
        try:
            ok = bool(answer) and float(answer) == float(row["right_answer"])
        except ValueError:
            ok = False
            unparsable += 1
        correct += ok
    n = len(dataset)
    print(f"{path}: {correct}/{n} = {100.0 * correct / n:.2f}%  (unparsable answers: {unparsable})")


if __name__ == "__main__":
    main()
