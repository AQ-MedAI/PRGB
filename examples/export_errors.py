#!/usr/bin/env python3
"""
Export Error Samples Example

This script demonstrates how to export inference error samples from evaluation results
for further analysis and debugging.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.eval_types import EvalResults


def export_error_samples(input_file, output_file):
    """
    Export error samples from evaluation results.
    
    Args:
        input_file (str): Path to the evaluation result file
        output_file (str): Path to save error samples
    """
    try:
        # Load evaluation results
        print(f"Loading evaluation results from: {input_file}")
        results = EvalResults.load_from_jsonl(input_file)
        
        # Get incorrect results
        errors = results.get_incorrect_results()
        print(f"Found {len(errors)} error samples out of {len(results.results)} total samples")
        
        # Export error samples
        results.save_to_jsonl(output_file, error_only=True)
        print(f"Error samples exported to: {output_file}")
        
        # Print some statistics
        print("\nError Analysis:")
        print(f"- Total samples: {len(results.results)}")
        print(f"- Correct samples: {len(results.results) - len(errors)}")
        print(f"- Error samples: {len(errors)}")
        print(f"- Accuracy: {(len(results.results) - len(errors)) / len(results.results) * 100:.2f}%")
        
        # Show first few error samples
        if errors:
            print("\nFirst 3 error samples:")
            for i, error in enumerate(errors[:3]):
                print(f"\nError {i+1}:")
                print(f"  ID: {error.id}")
                print(f"  Query: {error.query[:100]}...")
                print(f"  Expected: {error.answer[:100]}...")
                print(f"  Predicted: {error.prediction[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Error exporting samples: {e}")
        return False


def main():
    """Main function for error sample export."""
    
    # Example usage
    input_file = "results/Qwen3_eval_result.jsonl"  # Update this path
    output_file = "results/Qwen3_error_samples.jsonl"
    
    print("=== PRGB Error Sample Export ===")
    print("This script exports inference error samples for analysis.")
    print()
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        print("Please run evaluation first or update the input_file path.")
        return False
    
    # Export error samples
    success = export_error_samples(input_file, output_file)
    
    if success:
        print("\n✅ Error sample export completed successfully!")
        print(f"📁 Check the error samples in: {output_file}")
    else:
        print("\n❌ Error sample export failed!")
    
    return success


if __name__ == "__main__":
    main() 