from dataclasses import dataclass, asdict
from typing import List, Dict, Literal, Tuple
import json
import random
import logging

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class PlaceholderItem:
    placeholders: List[Dict[str, str]]
    answer: List[str]

    def __post_init__(self):
        if len(self.placeholders) != len(self.answer):
            raise ValueError(
                "The length of `placeholders` and `answer` must be the current length:"
                f"{len(self.placeholders)} and {len(self.answer)}"
            )

    def __len__(self):
        return len(self.placeholders)

    def __getitem__(self, i: int):
        return (self.placeholders[i], self.answer[i])

    @classmethod
    def from_dict(cls, data: Dict) -> "PlaceholderItem":
        try:
            return cls(
                placeholders=data["placeholders"], answer=data["answer"]
            )
        except KeyError as e:
            raise KeyError(f"Missing required field in PlaceholderItem: {e}")
        except TypeError:
            raise TypeError(
                f"Invalid data type in PlaceholderItem. Expected dict, got {type(data)}. Data: {data}"
            )


@dataclass
class RagData:
    id: str
    query: str
    golden_doc: List[str]
    noise_doc_level1: List[str]
    noise_doc_level2: List[str]
    noise_doc_level3: List[str]
    placeholder_item: List[PlaceholderItem]

    def __call__(
        self,
        st: Literal[
            "id",
            "query",
            "golden_doc",
            "noise_doc_level1",
            "noise_doc_level2",
            "noise_doc_level3",
            "placeholder_item",
        ],
    ):
        return asdict(self).get(st)

    @classmethod
    def from_dict(cls, data: Dict) -> "RagData":
        return cls(
            id=data["id"],
            query=data["query"],
            golden_doc=data["golden_doc"],
            noise_doc_level1=data["noise_doc_level1"],
            noise_doc_level2=data["noise_doc_level2"],
            noise_doc_level3=data["noise_doc_level3"],
            placeholder_item=PlaceholderItem.from_dict(
                data["placeholder_item"]
            ),
        )

    @classmethod
    def to_jsonl(cls, data_list: List["RagData"], file_path: str) -> None:
        """将RagData对象列表保存为JSONL文件

        Args:
            data_list: RagData对象列表
            file_path: 保存的JSONL文件路径
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for data in data_list:
                json_line = json.dumps(asdict(data), ensure_ascii=False)
                f.write(json_line + "\n")
        logger.info(
            f"Successfully \033[35mwrite {len(data_list)} benchmark items\033[0m to \033[32m{file_path}\033[0m"
        )

    @classmethod
    def from_jsonl(cls, file_path: str) -> List["RagData"]:
        """从JSONL文件读取数据并转换为RagData对象列表"""
        data_list = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # 跳过空行
                    try:
                        data_dict = json.loads(line)
                        data_list.append(cls.from_dict(data_dict))
                    except json.JSONDecodeError as e:
                        breakpoint()
                        print(f"Error parsing JSON at line {line_num}: {e}")
                        print(f"Line content: {line.strip()}")
                    except Exception as e:
                        breakpoint()
                        print(
                            f"Error processing line {line_num}: {type(e)}:{str(e)}"
                        )
                        print(f"Line content: {line.strip()}")
        logger.info(
            f"Successfully \033[34mloaded {len(data_list)} benchmark items\033[0m from \033[31m{file_path}\033[0m"
        )
        return data_list


class DataPreprocess:
    def __init__(
        self,
        data_path: str,
        prompt_config_path: str = "config/default_prompt_config.json",
    ):
        self.data: List[RagData] = RagData.from_jsonl(data_path)
        self.idx_hash = {}
        for i, dt in enumerate(self.data):
            self.idx_hash[dt.id] = i

        self.set_prompt_config(prompt_config_path)

    def __getitem__(self, i: int):
        return self.data[i]

    def __call__(self, idx: str):
        return self[self.idx_hash[idx]]

    def generate_input(
        self,
        num_iterations: int,
        noise_config: Dict[str, int] = {
            "noise_doc_level1": 1,
            "noise_doc_level2": 1,
            "noise_doc_level3": 1,
        },
        shuffle: bool = True,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Generate input for the model.
        Args:
            num_iterations: number of iterations to randomly select from available placeholders
            noise_config: noise configuration, it contains the number of noisy passages for each noise level
            shuffle: whether to shuffle the data
        Returns:
            idxs: list of indices
            queries: list of queries
            prompts: list of prompts
            answers: list of answers
        """
        queries_final = []
        prompts_final = []
        answers_final = []
        idxs_final = []
        for sample in self.data:
            idx = sample.id
            query = sample.query
            # Get total number of available placeholders
            total_placeholders = len(sample.placeholder_item.placeholders)
            # Check if num_iterations is valid
            if num_iterations > total_placeholders:
                # breakpoint()
                # raise ValueError(
                #     "`num_iterations` should be no more than the number of placeholders: "
                #     f"{total_placeholders}"
                # )
                # num_iterations = total_placeholders
                selected_indices = self.selection_rng.sample(
                    range(total_placeholders), total_placeholders
                )
            # Randomly select iteration indices using selection_rng
            else:
                selected_indices = self.selection_rng.sample(
                    range(total_placeholders), num_iterations
                )

            for cur_iter_idx in selected_indices:
                answers, golden_docs_ready = self.generate_golden_docs(
                    sample, cur_iter_idx
                )
                noise_docs_ready = self.generate_noise_docs(
                    sample, noise_config
                )
                docs_ready = golden_docs_ready + noise_docs_ready
                if shuffle:
                    self.shuffle_rng.shuffle(docs_ready)
                prompt = self.generate_prompt_cn(query, docs_ready)
                prompts_final.append(prompt)
                answers_final.append(answers)
                queries_final.append(query)
                idxs_final.append(idx)

        return idxs_final, queries_final, prompts_final, answers_final

    def set_prompt_config(self, prompt_config_path: str):
        with open(prompt_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            self.prompt_config = config
            # Initialize random number generators with seed from config
            random_seed = config.get("random_seed")
            self.selection_rng = random.Random(random_seed)
            self.shuffle_rng = random.Random(random_seed)

    def generate_golden_docs(
        self, sample: RagData, cur_iter_idx: int
    ) -> Tuple[str, List[str]]:
        """
        Generate golden docs for the sample by replacing the placeholder.
        Args:
            sample: RagData, sample
            cur_iter_idx: int, current iteration index
        Returns:
            answer: str, answer of the current iteration
            golden_docs_ready: List[str], golden docs of the current iteration
        """
        golden_docs_ready = []

        for gold_doc in sample.golden_doc:
            gold_doc_res = gold_doc
            for k, v in sample.placeholder_item.placeholders[
                cur_iter_idx
            ].items():
                gold_doc_res = gold_doc_res.replace(k, v)
            golden_docs_ready.append(gold_doc_res)
        return sample.placeholder_item.answer[cur_iter_idx], golden_docs_ready

    def generate_noise_docs(
        self, sample: RagData, noise_config: Dict[str, int]
    ) -> List[str]:
        """
        Generate noise docs for the sample.
        """
        noise_docs_ready = []
        for k, v in noise_config.items():
            noise_docs = sample(k)
            max_nums = min(len(noise_docs), v)
            noise_docs_ready.extend(
                self.selection_rng.sample(noise_docs, max_nums)
            )
        return noise_docs_ready

    def generate_prompt_cn(
        self, query: str, docs: List[str]
    ) -> List[Dict[str, str]]:
        """Format the default prompt for the model."""
        docs = "\n".join(
            [f"<doc_{i+1}>" + doc + "</doc>" for i, doc in enumerate(docs)]
        )
        return [
            {
                "role": "system",
                "content": self.prompt_config["system_prompt"],
            },
            {
                "role": "user",
                "content": self.prompt_config["user_prompt"].format(
                    docs=docs, query=query
                ),
            },
        ]


class CustomDataPreprocess(DataPreprocess):
    def __init__(self, data_path, prompt_config_path):
        super().__init__(data_path, prompt_config_path)


# 使用示例
if __name__ == "__main__":
    # 从JSONL文件读取数据
    data_list = RagData.from_jsonl(
        "/medrag/xingyi/data/placeholder/deductive_v2_fix_0514.jsonl"
    )

    # 处理读取到的数据
    for data in data_list[35:36]:
        print(f"Query: {data.query}")
        print(f"Golden docs: {data.golden_doc}")
        print(
            f"Placeholders: {[(data.placeholder_item[i][0], data.placeholder_item[i][1]) for i in range(len(data.placeholder_item))]}"
        )
        print("-" * 50)
