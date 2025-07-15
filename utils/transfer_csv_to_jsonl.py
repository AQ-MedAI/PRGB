import pandas as pd
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    return parser.parse_args()


def reformate_placeholder(x):
    placeholder = x["placeholder"]
    res = []
    for k, v in placeholder.items():
        if len(res) == 0:
            res = [{} for _ in range(len(v))]
        for i, val in enumerate(v):
            res[i][k] = val
    return {"answer": eval(x["answer"]), "placeholders": res}


def transfer_csv_to_jsonl(csv_path, jsonl_path):
    df = pd.read_csv(csv_path)

    df['noise_doc_level3'] = '[]'
    for col in [
        "placeholder",
        "golden_doc",
        "noise_doc_level1",
        "noise_doc_level2",
        "noise_doc_level3",
    ]:
        df[col] = df[col].apply(eval)

    df["placeholder_item"] = df.apply(reformate_placeholder, axis=1)
    df = df.drop("placeholder", axis=1)
    df = df.drop("answer", axis=1)
    # breakpoint()
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            json_line = json.dumps(row_dict, ensure_ascii=False)
            f.write(json_line + "\n")
    print(
        f"Successfully transferred \033[32m{csv_path}\033[0m to \033[31m{jsonl_path}\033[0m"
    )


if __name__ == "__main__":
    args = get_args()
    transfer_csv_to_jsonl(args.csv_path, args.jsonl_path)
