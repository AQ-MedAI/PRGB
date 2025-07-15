#!/usr/bin/env python3
"""
Basic RAG Evaluation Example

This script demonstrates how to use PRGB for basic RAG system evaluation.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import get_eval
import argparse
import json


def run_basic_evaluation():
    """Run a basic evaluation example."""
    
    # Example configuration
    config = {
        "model_name": "Qwen3",
        "model_path": "/path/to/your/model",  # Update this path
        "data_path": "tests/test.jsonl",
        "output_path": "./results",
        "num_iterations": 1,
        "noise_config": '{"noise_doc_level1":4,"noise_doc_level2":4,"noise_doc_level3":1}',
        "shuffle": True,
        "batch_size": 8,
        "temperature": 0.7
    }
    
    # Create a simple argument parser to simulate command line arguments
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**config)
    
    print("Starting basic RAG evaluation...")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_path}")
    
    try:
        # Run evaluation
        get_eval(args)
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return False
    
    return True


def run_custom_evaluation():
    """Run a custom evaluation with different parameters."""
    
    # Custom configuration for different scenarios
    scenarios = [
        {
            "name": "High Temperature",
            "config": {
                "model_name": "Qwen3",
                "model_path": "/path/to/your/model",
                "data_path": "tests/test.jsonl",
                "output_path": "./results/high_temp",
                "temperature": 0.9,
                "batch_size": 4
            }
        },
        {
            "name": "Low Temperature",
            "config": {
                "model_name": "Qwen3", 
                "model_path": "/path/to/your/model",
                "data_path": "tests/test.jsonl",
                "output_path": "./results/low_temp",
                "temperature": 0.3,
                "batch_size": 4
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\nRunning {scenario['name']} evaluation...")
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        args = Args(**scenario['config'])
        
        try:
            get_eval(args)
            print(f"{scenario['name']} evaluation completed!")
        except Exception as e:
            print(f"{scenario['name']} evaluation failed: {e}")


def main():
    """Main function for argument parsing and execution."""
    parser = argparse.ArgumentParser(description="PRGB Basic Example")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "custom"],
        default="basic",
        help="Evaluation mode: basic or custom"
    )
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        success = run_basic_evaluation()
    else:
        success = run_custom_evaluation()
    
    return 0 if success else 1


if __name__ == "__main__":
    main() 