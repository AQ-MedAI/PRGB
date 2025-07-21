import re
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..logger import get_logger

logger = get_logger()

# vLLM imports - only imported when this module is used
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    raise ValueError("vLLM is not installed. Please install it with: pip install vllm")

def transfer_dict_conv(
    inputs: List[str], system: Optional[str] = None
) -> List[Dict[str, str]]:
    assert len(inputs) % 2 == 1, "number of rounds must be odd"
    output_chat_dict = (
        [{"role": "system", "content": system}] if system else []
    )

    for i in range(len(inputs)):
        if i % 2 == 0:
            output_chat_dict.append({"role": "user", "content": inputs[i]})
        elif i % 2 == 1:
            output_chat_dict.append(
                {"role": "assistant", "content": inputs[i]}
            )
    return output_chat_dict

class CommonModelVllm:
    def __init__(
        self, plm="/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct"
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
        
        self.model = LLM(
            model=plm,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=8192,
            enable_prefix_caching=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        self.tokenizer = self.model.get_tokenizer()

    def batch_generate(
        self, data, temperature=0.0, system="", top_p=0.8, batch_size=16
    ):
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=800
        )
        if isinstance(data[0], str):
            data = list(map(self.process_special_token, range(len(data))))
        elif isinstance(data[0], list):
            data = self.tokenizer.apply_chat_template(
                data, tokenize=False, add_generation_prompt=True
            )
        else:
            raise ValueError(
                "data must be a list of strings or a list of lists"
            )

        generate_result = []

        for i in tqdm(range(0, len(data), batch_size)):
            model_inputs = data[i : i + batch_size]
            generated_ids = self.model.generate(model_inputs, sampling_params)

            for output in generated_ids:
                generate_result.append(output.outputs[0].text)
            torch.cuda.empty_cache()
            if i == 0:
                logger.info(f"First generated result: {generate_result[0]}")
        return generate_result

    def single_generate(self, prompt):
        model_inputs = self.tokenizer.apply_chat_template(
            transfer_dict_conv([prompt]),
            tokenize=False
        )
        # logger.debug("=" * 20, "Direct Ans", "=" * 20)
        generated_ids = self.model.generate(
            model_inputs, self.sampling_params
        )
        return self.tokenizer.decode(generated_ids[0].outputs[0].text)

    def process_special_token(self, text, system):
        return self.tokenizer.apply_chat_template(
            transfer_dict_conv([text], system),
            tokenize=False,
            add_generation_prompt=True,
        )


class InferModelVllm(CommonModelVllm):
    def batch_generate(
        self, data, temperature=0.0, system="", top_p=0.8, batch_size=16
    ):
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=800
        )
        if isinstance(data[0], str):
            data = list(map(self.process_special_token, range(len(data))))
        elif isinstance(data[0], list):
            data = self.tokenizer.apply_chat_template(
                data, tokenize=False, add_generation_prompt=True
            )
        else:
            raise ValueError(
                "data must be a list of strings or a list of lists"
            )

        generate_result = []

        for i in tqdm(range(0, len(data), batch_size)):
            model_inputs = data[i : i + batch_size]
            generated_ids = self.model.generate(model_inputs, sampling_params)

            for output in generated_ids:
                generate_result.append(output.outputs[0].text)
            torch.cuda.empty_cache()
            if i == 0:
                logger.info(f"First generated result: {generate_result[0]}")
        return generate_result

    def extract_anwer(self, answer):
        return re.sub(r"<think>.*</think>", "", answer, flags=re.DOTALL).strip()


class Qwen3Vllm(CommonModelVllm):
    def __init__(self, plm, think_mode):
        super().__init__(plm)
        self.think_mode = think_mode

    def batch_generate(
        self, data, temperature=0.0, system="", top_p=0.8, batch_size=16
    ):
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=800
        )
        if isinstance(data[0], str):
            data = list(map(self.process_special_token, range(len(data))))
        elif isinstance(data[0], list):
            data = self.tokenizer.apply_chat_template(
                data, tokenize=False, add_generation_prompt=True, enable_thinking=self.think_mode
            )
        else:
            raise ValueError(
                "data must be a list of strings or a list of lists"
            )

        generate_result = []

        for i in tqdm(range(0, len(data), batch_size)):
            model_inputs = data[i : i + batch_size]
            generated_ids = self.model.generate(model_inputs, sampling_params)

            for output in generated_ids:
                generate_result.append(output.outputs[0].text)
            torch.cuda.empty_cache()
            if i == 0:
                logger.info(f"First generated result: {generate_result[0]}")
        return generate_result


class HiragVllm(CommonModelVllm):
    def __init__(self, plm, think_mode):
        super().__init__(plm)
        self.think_mode = think_mode

    def batch_generate(
        self, data, temperature=0.0, system="", top_p=0.8, batch_size=16
    ):
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=800
        )
        if isinstance(data[0], str):
            data = list(map(self.process_special_token, range(len(data))))
        elif isinstance(data[0], list):
            data = self.tokenizer.apply_chat_template(
                data, tokenize=False, add_generation_prompt=True, add_think_prompt=self.think_mode
            )
        else:
            raise ValueError(
                "data must be a list of strings or a list of lists"
            )

        generate_result = []

        for i in tqdm(range(0, len(data), batch_size)):
            model_inputs = data[i : i + batch_size]
            generated_ids = self.model.generate(model_inputs, sampling_params)

            for output in generated_ids:
                generate_result.append(output.outputs[0].text)
            torch.cuda.empty_cache()
            if i == 0:
                logger.info(f"First generated result: {generate_result[0]}")
        return generate_result 