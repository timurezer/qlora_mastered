import argparse
import json
import pandas as pd
import numpy as np

"""
Convert dataset to alpaca-style dataset for Mindspeed-LLM training
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to train dataset json file")
    parser.add_argument("--save_path", type=str, help="path to new parquet file")
    parser.add_argument("--seed", type=int, default=42, help="seed to shuffle inputs")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    role_mapping={
        "system": "text",
        "user": "instruction",
        "assistant": "output"
    }

    df = []

    np.random.seed(args.seed)
    idx = np.random.permutation(len(data))

    for i in idx:
        row = data[i]
        sub_df = {}
        for d in row["messages"]:
            new_role = role_mapping[d["role"]]
            sub_df[new_role] = d["content"]
        
        sub_df["input"] = ""
        df.append(sub_df)

    df = pd.DataFrame(df)
    print(df)

    df.to_parquet(args.save_path)
    print(f"Saved to {args.save_path}")


main()

