import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Literal

import pandas as pd
from tqdm import tqdm

from .data import DataPreprocess
from .models import Qwen3Vllm


def checkanswer_acc(
    predictions: List[str], answers: List[str]
) -> Tuple[List[int], List[int]]:
    labels = []

    def evaluate_expression(expr: str, prediction: str) -> bool:
        # If the expression doesn't contain any logical operators, do direct text matching
        if not any(op in expr for op in ["&", "|", "(", ")"]):
            return expr.strip() in prediction

        # Tokenize the expression
        tokens = []
        current = ""
        for char in expr:
            if char in ["&", "|", "(", ")"]:
                if current:
                    tokens.append(current.strip())
                    current = ""
                tokens.append(char)
            else:
                current += char
        if current:
            tokens.append(current.strip())

        # Convert to postfix notation (Reverse Polish Notation)
        def precedence(op):
            if op == "&":
                return 2
            if op == "|":
                return 1
            return 0

        output = []
        operators = []

        for token in tokens:
            if token in ["&", "|"]:
                while (
                    operators
                    and operators[-1] != "("
                    and precedence(operators[-1]) >= precedence(token)
                ):
                    output.append(operators.pop())
                operators.append(token)
            elif token == "(":
                operators.append(token)
            elif token == ")":
                while operators and operators[-1] != "(":
                    output.append(operators.pop())
                if operators and operators[-1] == "(":
                    operators.pop()
            else:
                output.append(token)

        while operators:
            output.append(operators.pop())

        # Evaluate the postfix expression
        stack = []
        for token in output:
            if token in ["&", "|"]:
                b = stack.pop()
                a = stack.pop()
                if token == "&":
                    stack.append(a and b)
                else:  # token == '|'
                    stack.append(a or b)
            else:
                # Check if the term exists in prediction
                stack.append(token in prediction)

        return stack[0] if stack else False

    errors = []
    for i, (predict, answer) in enumerate(zip(predictions, answers)):
        label = 1

        if not evaluate_expression(answer, predict):
            label = 0
            errors.append(i)
        labels.append(label)
    return errors, labels


def save_to_jsonl(
    idx_list: List[str],
    query_list: List[str],
    prompt_list: List[str],
    answer_list: List[str],
    prediction_list: List[str],
    label_list: List[Any],
    output_path: str,
    append: bool = False,
) -> None:
    # 检查输入
    lists = [
        idx_list,
        query_list,
        prompt_list,
        answer_list,
        prediction_list,
        label_list,
    ]
    lengths = [len(lst) for lst in lists]
    if len(set(lengths)) != 1:
        raise ValueError(f"Lists have different lengths: {lengths}")

    # 确定写入模式
    mode = "a" if append else "w"

    try:
        with open(output_path, mode, encoding="utf-8") as f:
            # 使用tqdm显示进度
            for items in tqdm(
                zip(*lists), total=len(query_list), desc="Saving to JSONL"
            ):
                idx, query, prompt, answer, prediction, label = items

                # 创建数据字典
                data = {
                    "id": idx,
                    "query": query,
                    "prompt": prompt,
                    "answer": answer,
                    "prediction": prediction,
                    "label": label,
                }

                try:
                    # 写入一行
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error writing line: {e}")
                    print(f"Problematic data: {data}")
                    continue

    except Exception as e:
        print(f"Error opening file {output_path}: {e}")
        raise

    print(f"Successfully saved {len(query_list)} items to {output_path}")


def get_eval(args):
    model_name = args.model_name
    model_path = args.model_path
    output_path = args.output_path
    noise_config = json.loads(args.noise_config)
    shuffle = args.shuffle
    batch_size = args.batch_size
    temperature = args.temperature

    ragdata = DataPreprocess(args.data_path)

    # 采样
    ragdata.data = ragdata.data[:5]

    if "Qwen" in model_name:
        model = Qwen3Vllm(plm=model_path)  # TODO 会报错，非qwen
    results = {}
    prompts = []
    answers = []
    queries = []
    idxs = []
    idx, query, prompt, answer = ragdata.generate_input(
        1, shuffle=shuffle, noise_config=noise_config
    )
    prompts.extend(prompt)
    answers.extend(answer)
    queries.extend(query)
    idxs.extend(idx)
    predictions = model.batch_generate(
        prompts, temperature, batch_size=batch_size
    )
    error, labels = checkanswer_acc(predictions, answers)
    results["prompt"] = prompts
    results["ground_truth"] = answers
    results["predictions"] = predictions
    results["label"] = labels
    results["query"] = queries
    results["id"] = idxs
    breakpoint()

    save_to_jsonl(
        idxs,
        queries,
        prompts,
        answers,
        predictions,
        labels,
        f"{output_path}/{model_name}_eval_result.jsonl",
    )

    error_id = [idxs[i] for i in error]
    results = pd.DataFrame(results)
    scores = results.groupby("id")["label"].mean().mean()
    with open(
        f"{output_path}/{model_name}_eval_scores.jsonl", "w", encoding="utf-8"
    ) as f:
        tmp = {"acc_scores": scores, "error_id": error_id}
        json.dump(tmp, f, ensure_ascii=False)

    print("acc_scores:", scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name", type=str, default="Qwen", help="model name"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="tests/test.jsonl",
        help="evaluetion dataset",
    )
    parser.add_argument(
        "--model-path", type=str, default="", help="api key of chatgpt"
    )
    parser.add_argument(
        "--nosie_passages_num", type=int, default=3, help="name of plm"
    )
    parser.add_argument(
        "--output-path", type=str, default="./", help="url of chatgpt"
    )
    parser.add_argument(
        "--noise_config",
        type=str,
        default='{"noise_doc_level1":3,"noise_doc_level2":3,"noise_doc_level3":0}',
        help="corpus id",
    )
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="rate of noisy passages"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="rate of correct passages"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="number of external passages",
    )
    args = parser.parse_args()
    get_eval(args)
