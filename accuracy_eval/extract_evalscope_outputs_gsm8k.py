# extract_evalscope_outputs_gsm8k.py
import argparse
import json
import os
import re

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()


def extract_boxed_text(input_string):
    pattern = r'\\boxed{([^}]+)}'
    matched = re.findall(pattern, input_string)
    if matched:
        return matched[-1]
    
    return ""



def extract_and_save(path: str, data_path: str, save_path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        raw_file = [json.loads(x) for x in f]

    answers = dict()
    reasoning = dict()
    for item in raw_file:
        answer = item["model_output"]["choices"][0]["message"]["content"]
        if isinstance(answer, str):
            answers[item["index"]] = answer
        elif isinstance(answer, list):
            reasoning[item["index"]] = answer[0]["reasoning"]
            # try:
            #     reasoning[item["index"]] = answer[0]["reasoning"]
            # except:
            #     # print(f"item {item['index']} have not finished reasoning")
            #     raise
            answers[item["index"]] = answer[1]["text"]

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []
    for i in list((answers.keys())):
        predicted_value = extract_boxed_text(answers[i])
        data[i]["output"] = predicted_value
        data[i]["pred_output"] = answers[i]
        if reasoning:
            try:
                data[i]["pred_reasoning"] = reasoning[i]
            except:
                print(f"item {i} have not finished reasoning")
        output.append(data[i])

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Saved to {save_path}")


def main() -> None:
    args = parse_args()
    if args.save_path is None:
        args.save_path = args.path

    if os.path.isdir(args.path):
        for folder, _, files in os.walk(args.path):
            # format: <folder>/<date_time>/predictions/lora-adapter [] ['general_qa_test_data.jsonl']
            if (
                "predictions" in folder
                and len(files) == 1
                and files[0].endswith(".jsonl")
            ):
                print(f"Found in {folder} files: {files}")
                ans_file_path = os.path.join(folder, files[0])
                date_time, model_name = folder.split("/")[-3], folder.split("/")[-1]
                new_file_name = f"results_{date_time}_{model_name}.json"
                save_file_path = os.path.join(args.save_path, new_file_name)
                extract_and_save(ans_file_path, args.data_path, save_file_path)
    else:
        extract_and_save(args.path, args.data_path, args.save_path)


if __name__ == "__main__":
    main()