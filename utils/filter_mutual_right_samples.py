import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data import DataPreprocess, RagData
from core.eval_types import EvalResults
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

dt = RagData.from_jsonl(
    "/medrag/xingyi/data/placeholder/ad_compare_0327_fix.jsonl"
)

gpt4o = EvalResults.load_from_jsonl(
    "/medrag/xingyi/data/placeholder/gpt-4o_ad_compare_0327_fix_eval_result.jsonl"
)
gpt4 = EvalResults.load_from_jsonl(
    "/medrag/xingyi/data/placeholder/gpt/gpt_eval_result_adcompare.jsonl"
)
qwen_max = EvalResults.load_from_jsonl(
    "/medrag/xingyi/data/placeholder/qwen_ad_compare_0327_fix_eval_result.jsonl"
)
error_ids = gpt4o.error_ids + gpt4.error_ids + qwen_max.error_ids
error_collection = [
    item for item, count in Counter(error_ids).items() if count == 2
]

ad_compare_ids = [
    "compare_ad-368",
    "compare_ad-377",
    "compare_ad-380",
    "compare_ad-386",
    "compare_ad-391",
    "compare_ad-392",
    "compare_ad-393",
    "compare_ad-405",
    "compare_ad-406",
    "compare_ad-407",
    "compare_ad-412",
    "compare_ad-414",
    "compare_ad-417",
    "compare_ad-420",
    "compare_ad-431",
]
for human_fix_id in ad_compare_ids:
    if human_fix_id in error_collection:
        error_collection.remove(human_fix_id)

dt_fixed = [d for d in dt if d.id not in error_collection]
RagData.to_jsonl(
    dt_fixed, "/medrag/xingyi/data/placeholder/ad_compare_0515_fix.jsonl"
)
