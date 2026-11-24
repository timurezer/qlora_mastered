# get_acc.py
import argparse
import json
import os
import re

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help=(
            "Path to the JSON dataset file or folder with files. "
            "Takes files in format result*.json"
        ),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="acc.xlsx",
        help="Path to save the Excel output file",
    )
    parser.add_argument(
        "--nums_compare",
        action="store_true"
    )


    return parser.parse_args()


def preprocess_answer(x: str) -> str:
    output = x.replace("【", "").replace("】", "").strip()
    return output


def process_file(path: str, args) -> dict[str, str | float]:
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    is_correct = []  # overall correctness
    lang_correct = {}  # language -> list of correct booleans

    for row in dataset:
        answer = preprocess_answer(row["output"])
        if args.nums_compare:
            # clear and store only digits, minus and floating point
            answer = re.sub(r"[^\d.-]", "", answer)
            if not answer:
                correct = False
            else:
                correct = float(answer) == float(row["right_answer"])
        else:
            correct = answer == row["right_answer"]
        is_correct.append(correct)

        lang = row.get("language", "unknown")
        if lang not in lang_correct:
            lang_correct[lang] = []
        lang_correct[lang].append(correct)

    # Overall accuracy
    overall_acc = sum(is_correct) / len(is_correct)
    print(f"Overall accuracy: {overall_acc:.4f}")

    # Per-language accuracy
    lang_acc = {"file_name": os.path.basename(path), "total": overall_acc}

    for lang, corr_list in lang_correct.items():
        lang_acc[lang] = sum(corr_list) / len(corr_list)
        print(
            f"Accuracy for {lang}: {lang_acc[lang]:.4f} ({sum(corr_list)}/{len(corr_list)})"
        )

    return lang_acc


def main() -> None:
    args = parse_args()
    if os.path.isfile(args.path):
        files = [args.path]
    else:
        print(f"Find all results*.json files at the {args.path}")
        files = [x for x in os.listdir(args.path) if ".json" in x and "results" in x]
        files = [os.path.join(args.path, x) for x in files]

    df = []
    for file_path in files:
        acc = process_file(file_path, args)
        df.append(acc)

    df = pd.DataFrame(df).sort_values(by=["file_name"], axis=0)
    print(df)
    df.to_excel(args.save_path, index=False)


if __name__ == "__main__":
    main()
    