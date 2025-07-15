#!/usr/bin/env python3
"""
PRGB - Main Evaluation Script

This script provides the main entry point for RAG system evaluation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core import get_eval

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function for RAG evaluation."""
    parser = argparse.ArgumentParser(
        description="PRGB - RAG System Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with Qwen3 model
  python eval.py --model-name "Qwen3" --model-path "/path/to/model" --data-path "data/test.jsonl"
  
  # Evaluation with custom noise configuration
  python eval.py --model-name "Qwen3" --noise-config '{"noise_doc_level1":4,"noise_doc_level2":4,"noise_doc_level3":1}'
  
  # Batch evaluation with specific parameters
  python eval.py --model-name "Qwen3" --batch-size 32 --temperature 0.8 --shuffle True
        """
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen3",
        help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model or API key"
    )

    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        default="tests/test.jsonl",
        help="Path to the evaluation dataset"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of evaluation iterations. For each query, randomly select n different placeholders to run evaluation. Each placeholder represents a different version of the same query with different variable substitutions."
    )

    # Output configuration
    parser.add_argument(
        "--output-path",
        type=str,
        default="./results",
        help="Output directory for results"
    )

    # Evaluation parameters
    parser.add_argument(
        "--noise-config",
        type=str,
        default='{"noise_doc_level1":4,"noise_doc_level2":4,"noise_doc_level3":1}',
        help="Noise configuration as JSON string"
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation"
    )
    parser.add_argument(
        "--custom_config",
        type=str,
        default=None,
        help="custom prompt config path",
    )
    # Additional options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    try:
        noise_config = json.loads(args.noise_config)
    except json.JSONDecodeError:
        logger.error("Invalid noise_config JSON format")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation with model: {args.model_name}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Noise config: {noise_config}")

    try:
        # Run evaluation
        get_eval(args)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 