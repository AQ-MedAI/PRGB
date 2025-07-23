import argparse
import json
from typing import List, Tuple

from .data import DataPreprocess
from .eval_types import EvalResult, EvalResults
from .logger import get_logger

logger = get_logger()


def checkanswer_acc(
    predictions: List[str], answers: List[str], is_infer_model: bool = False
) -> Tuple[List[int], List[int]]:
    if is_infer_model:
        import re
        predictions = [re.sub(r"<think>.*</think>", "", answer, flags=re.DOTALL).strip() for answer in predictions]

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


def get_eval(args):
    model_name = args.model_name
    model_path = args.model_path
    output_path = args.output_path
    noise_config = json.loads(args.noise_config)
    shuffle = args.shuffle
    batch_size = args.batch_size
    temperature = args.temperature

    if args.custom_config:
            ragdata = DataPreprocess(
                args.data_path, args.custom_config
            )
    else:
        if "zh" in args.data_path:
            ragdata = DataPreprocess(
                args.data_path, "config/api_prompt_config_ch.json"
            )
        elif "en" in args.data_path:
            ragdata = DataPreprocess(
                args.data_path, "config/api_prompt_config_en.json"
            )

    # 采样
    # ragdata.data = ragdata.data[:1]
    if "http" in model_path:
        import importlib.util
        if importlib.util.find_spec("openai") is None:
            from .models import APIModel
            model = APIModel(url=model_path, model=model_name, api_key=args.api_key, inference_mode=args.inference_mode)
        else:
            from .models import OpenAIModel
            model = OpenAIModel(url=model_path, model=model_name, api_key=args.api_key, inference_mode=args.inference_mode)

    else:
        if not args.inference_mode:
            if "qwen3" in model_name.lower():
                from .models import Qwen3Vllm
                model = Qwen3Vllm(plm=model_path, think_mode=False)
            else:
                from .models import CommonModelVllm
                model = CommonModelVllm(plm=model_path)
        else:
            from .models import InferModelVllm
            model = InferModelVllm(plm=model_path)

    prompts = []
    answers = []
    queries = []
    idxs = []
    idx, query, prompt, answer = ragdata.generate_input(
        args.num_iterations, shuffle=shuffle, noise_config=noise_config
    )
    prompts.extend(prompt)
    answers.extend(answer)
    queries.extend(query)
    idxs.extend(idx)
    predictions = model.batch_generate(
        prompts, temperature, batch_size=batch_size
    )
    error, labels = checkanswer_acc(predictions, answers, is_infer_model=args.inference_mode)

    eval_results = EvalResults()
    for i in range(len(idxs)):
        result = EvalResult(
            id=idxs[i],
            query=queries[i],
            prompt=prompts[i],
            answer=answers[i],
            prediction=predictions[i],
            label=labels[i],
        )
        eval_results.add_result(result)

    eval_results.calculate_scores(True)
    eval_results.save_to_jsonl(f"{output_path}/{model_name}_eval_result_{str(noise_config)}.jsonl")

    with open(
        f"{output_path}/{model_name}_eval_scores.jsonl", "w", encoding="utf-8"
    ) as f:
        json.dump(eval_results.to_dict(), f, ensure_ascii=False)
    print(f"Saving to {output_path}/{model_name}_eval_scores.jsonl")
    print("acc_scores:", eval_results.acc_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api-key", type=str, default=None, help="api key of api models"
    )
    parser.add_argument(
        "--model-name", type=str, default="Qwen", help="model name"
    )
    parser.add_argument(
        "--inference-mode", type=bool, default=False, help="whether inference model or not"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="tests/test.jsonl",
        help="evaluetion dataset",
    )
    parser.add_argument(
        "--model-path", type=str, default="", help="api key of api models or local model path"
    )
    parser.add_argument(
        "--nosie_passages_num", type=int, default=3, help="number of noisy passages"
    )
    parser.add_argument(
        "--output_path", type=str, default="./", help="output path"
    )
    parser.add_argument(
        "--custom_config",
        type=str,
        default=None,
        help="custom config path",
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
        "--num_iterations", type=int, default=3, help="Number of evaluation iterations. For each query, randomly select n different placeholders to run evaluation. Each placeholder represents a different version of the same query with different variable substitutions."
    )
    parser.add_argument(
        "--gpu", type=int, default=8, help="number of iterations"
    )
    args = parser.parse_args()
    get_eval(args)
