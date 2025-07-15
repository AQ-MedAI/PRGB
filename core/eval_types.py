import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Single evaluation result."""

    id: str
    query: str
    prompt: str
    answer: str
    prediction: str
    label: int
    rag_class: str = field(init=False)

    def __post_init__(self):
        """Initialize rag_class after other fields are set."""
        self.rag_class = self.id.split("-")[0]

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalResult":
        return cls(
            id=data["id"],
            query=data["query"],
            prompt=data["prompt"],
            answer=data["answer"],
            prediction=data["prediction"],
            label=data["label"],
        )


@dataclass
class EvalResults:
    """Collection of evaluation results."""

    results: List[EvalResult] = field(default_factory=list)
    acc_scores: float = 0.0
    acc_scores_by_rag_class: Dict[str, float] = field(default_factory=dict)
    error_ids: List[str] = field(default_factory=list)

    def __getitem__(self, idx: int) -> EvalResult:
        return self.results[idx]

    def __call__(self, result_id: str) -> EvalResult:
        """Get EvalResult by id.

        Args:
            result_id: ID of the result to find

        Returns:
            EvalResult with matching id

        Raises:
            KeyError: If no result found with given id
        """
        for result in self.results:
            if result.id == result_id:
                return result
        raise KeyError(f"No result found with id {result_id}")

    def add_result(self, result: EvalResult) -> None:
        self.results.append(result)

    @classmethod
    def load_from_jsonl(cls, input_path: str) -> "EvalResults":
        """Load results from a JSONL file.

        Args:
            input_path: Path to the JSONL file to load from

        Returns:
            EvalResults object containing the loaded results
        """
        eval_results = cls()

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                # Use tqdm to show progress
                for line in tqdm(f, desc="Loading from JSONL"):
                    try:
                        data = json.loads(line.strip())
                        result = EvalResult.from_dict(data)
                        eval_results.add_result(result)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON line: {e}")
                        logger.error(f"Problematic line: {line}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line: {e}")
                        logger.error(f"Problematic line: {line}")
                        continue

        except Exception as e:
            logger.error(f"Error opening file {input_path}: {e}")
            raise

        # Calculate scores after loading all results
        eval_results.calculate_scores(True)

        logger.info(
            f"Successfully loaded \033[34m{len(eval_results.results)} eval_results\033[0m from \033[31m{input_path}\033[0m"
        )

        return eval_results

    def calculate_scores(self, by_rag_class: bool = False) -> None:
        if not self.results:
            return

        # Calculate accuracy scores
        df = pd.DataFrame([vars(r) for r in self.results])

        if by_rag_class:
            # First calculate mean for each id
            self.id_scores = df.groupby("id")["label"].mean()
            # Convert to DataFrame with rag_class
            class_hash = {
                "combination_v1": "combination",
                "combination_v2": "combination",
                "combination_v3": "combination",
                "filter": "filter",
            }
            self.id_scores_df = pd.DataFrame(
                {
                    "score": self.id_scores,
                    "rag_class": df.groupby("id")["rag_class"].first(),
                }
            )
            self.id_scores_df['rag_class_all'] = self.id_scores_df['rag_class'].apply(lambda x: class_hash.get(x, "infer"))
            # Then group by rag_class and calculate mean
            class_scores = self.id_scores_df.groupby("rag_class")["score"].mean()
            class_scores_all = self.id_scores_df.groupby("rag_class_all")["score"].mean()
            self.acc_scores_by_rag_class = class_scores.to_dict()
            self.acc_scores_by_rag_class_all = class_scores_all.to_dict()
            logger.info("\nScores by RAG class:")
            for rag_class, score in class_scores_all.items():
                logger.info(
                    f"\033[34m{rag_class}:\033[0m \033[1m{score:.4f}\033[0m"
                )

        self.acc_scores = df.groupby("id")["label"].mean().mean()

        # Get error IDs
        self.error_ids = [r.id for r in self.results if r.label == 0]
        print(
            f"\033[41;37mOverall Acc_scores:\033[0m \033[1m{self.acc_scores:.4f}\033[0m"
        )

    def to_dict(self) -> Dict:
        return {"acc_scores": self.acc_scores, "error_ids": self.error_ids}

    def get_correct_results(self) -> List[EvalResult]:
        """Get all results that were correctly predicted (label == 1)."""
        return [r for r in self.results if r.label == 1]

    def get_incorrect_results(self) -> List[EvalResult]:
        """Get all results that were incorrectly predicted (label == 0)."""
        return [r for r in self.results if r.label == 0]

    def save_to_jsonl(
        self, output_path: str, append: bool = False, error_only: bool = False
    ) -> None:
        """Save all results to a JSONL file.

        Args:
            output_path: Path to save the JSONL file
            append: Whether to append to existing file or create new one
            error_only: If True, only save results with label == 0 (incorrect predictions)
        """
        if not self.results:
            logger.error("No results to save")
            return

        # Determine write mode
        mode = "a" if append else "w"

        # Filter results if error_only is True
        results_to_save = (
            self.get_incorrect_results() if error_only else self.results
        )

        if not results_to_save:
            logger.warning("No results to save after filtering")
            return

        try:
            with open(output_path, mode, encoding="utf-8") as f:
                # Use tqdm to show progress
                for result in results_to_save:
                    # Create data dictionary
                    data = {
                        "id": result.id,
                        "query": result.query,
                        "prompt": result.prompt,
                        "answer": result.answer,
                        "prediction": result.prediction,
                        "label": result.label,
                    }

                    try:
                        # Write one line
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logger.error(f"Error writing line: {e}")
                        logger.error(f"Problematic data: {data}")
                        continue

        except Exception as e:
            logger.error(f"Error opening file {output_path}: {e}")
            raise

        logger.info(
            f"Successfully saved {len(results_to_save)} items to \033[31m{output_path}\033[0m"
        )


if __name__ == "__main__":
    # 加载数据
    results = EvalResults.load_from_jsonl(
        "gpt_eval_result.jsonl"
    )
    errors = results.get_incorrect_results()
    results.save_to_jsonl(
        "gpt_eval_result_error.jsonl",
        error_only=True,
    )
