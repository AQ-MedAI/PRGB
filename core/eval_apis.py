import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Literal

from tqdm import tqdm
from .eval_types import EvalResult, EvalResults
from .data import DataPreprocess

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api_models.api import request_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

        if not evaluate_expression(answer.lower(), predict.lower()):
            label = 0
            errors.append(i)
        labels.append(label)
    return errors, labels


class EvalAPI:
    def __init__(self, args):
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.noise_config = json.loads(args.noise_config)
        self.shuffle = args.shuffle
        self.batch_size = args.batch_size
        self.temperature = args.temperature
        self.data_path = args.data_path
        self.num_iterations = args.num_iterations
        self.ragdata = DataPreprocess(
            args.data_path, "config/api_prompt_config_en.json"
        )
        self.eval_results = EvalResults()

    def generate_predictions(self, prompts):
        prompts = [prompt[1]["content"] for prompt in prompts]
        if "qwen" in self.model_name:
            predictions = request_qwen(prompts)
        elif "gpt" in self.model_name:
            predictions = request_gpt(prompts, model=self.model_name)
        return [x[2] for x in predictions]

    def generate_predictions_api(self, prompts):
        predictions = request_model(prompts, model=self.model_name)
        return [x[2] for x in predictions]

    def evaluate(self):
        prompts = []
        answers = []
        queries = []
        idxs = []

        idx, query, prompt, answer = self.ragdata.generate_input(
            self.num_iterations, shuffle=self.shuffle, noise_config=self.noise_config
        )
        prompts.extend(prompt)
        answers.extend(answer)
        queries.extend(query)
        idxs.extend(idx)

        predictions = self.generate_predictions_api(prompts)
        errors, labels = checkanswer_acc(predictions, answers)

        # Store results using EvalResult class
        for i in range(len(idxs)):
            result = EvalResult(
                id=idxs[i],
                query=queries[i],
                prompt=prompts[i],
                answer=answers[i],
                prediction=predictions[i],
                label=labels[i],
            )
            self.eval_results.add_result(result)

        # Save results and calculate scores
        self.eval_results.save_to_jsonl(
            f"{self.output_path}/{self.model_name}_{os.path.splitext(os.path.basename(self.data_path))[0]}_eval_result.jsonl"
        )
        self.eval_results.calculate_scores(by_rag_class=True)

        # Save scores
        with open(
            f"{self.output_path}/{self.model_name}_{os.path.splitext(os.path.basename(self.data_path))[0]}_eval_scores.jsonl",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.eval_results.to_dict(), f, ensure_ascii=False)

        print(
            f"Saving to {self.output_path}/{self.model_name}_{os.path.splitext(os.path.basename(self.data_path))[0]}_eval_scores.jsonl"
        )
        print("acc_scores:", self.eval_results.acc_scores)
        return self.eval_results.acc_scores


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
        "--nosie_passages_num", type=int, default=5, help="name of plm"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./eval_result/qwen",
        help="url of chatgpt",
    )
    parser.add_argument(
        "--noise_config",
        type=str,
        default='{"noise_doc_level1":4,"noise_doc_level2":4,"noise_doc_level3":1}',
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
    parser.add_argument(
        "--num_iterations", type=int, default=1, help="number of iterations"
    )
    args = parser.parse_args()

    # Create evaluator instance and run evaluation
    evaluator = EvalAPI(args)
    evaluator.evaluate()
