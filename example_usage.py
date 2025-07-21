#!/usr/bin/env python3
"""
Example usage of the separated models structure
"""

def example_api_usage():
    """Example of using API models"""
    print("=== API Models Example ===")
    
    try:
        from core.models import APIModel, OpenAIModel

        # Example 1: Using APIModel (requires API key)
        print("\n1. APIModel example:")
        try:
            # Set your API key here or use environment variable
            # os.environ["API_KEY"] = "your_api_key_here"
            
            api_model = APIModel(
                model="gpt-3.5-turbo",
                api_key="your_api_key_here"  # Replace with your actual API key
            )
            
            # Single generation
            response = api_model.generate(
                text="Hello, how are you?",
                temperature=0.7,
                system="You are a helpful assistant."
            )
            print(f"Response: {response}")
            
            # Batch generation
            texts = ["Hello", "How are you?", "What's the weather like?"]
            responses = api_model.batch_generate(
                data=texts,
                temperature=0.7,
                system="You are a helpful assistant.",
                qps=5  # 5 requests per second
            )
            print(f"Batch responses: {responses}")
            
        except Exception as e:
            print(f"APIModel error (expected without API key): {e}")
        
        # Example 2: Using OpenAIModel (requires OpenAI package)
        print("\n2. OpenAIModel example:")
        try:
            openai_model = OpenAIModel(
                model="gpt-3.5-turbo",
                api_key="your_api_key_here"  # Replace with your actual API key
            )
            
            response = openai_model.generate(
                text="Explain quantum computing in simple terms.",
                temperature=0.7
            )
            print(f"Response: {response}")
            
        except ImportError as e:
            print(f"OpenAIModel error (OpenAI not installed): {e}")
        except Exception as e:
            print(f"OpenAIModel error (expected without API key): {e}")
            
    except ImportError as e:
        print(f"Failed to import API models: {e}")

def example_vllm_usage():
    """Example of using vLLM models"""
    print("\n=== vLLM Models Example ===")
    
    try:
        from core.models import CommonModelVllm, HiragVllm, Qwen3Vllm

        # Example 1: Using CommonModelVllm
        print("\n1. CommonModelVllm example:")
        try:
            vllm_model = CommonModelVllm(
                plm="/path/to/your/model"  # Replace with your model path
            )
            
            # Single generation
            response = vllm_model.single_generate("Hello, how are you?")
            print(f"Response: {response}")
            
            # Batch generation
            texts = ["Hello", "How are you?", "What's the weather like?"]
            responses = vllm_model.batch_generate(
                data=texts,
                temperature=0.7,
                batch_size=4
            )
            print(f"Batch responses: {responses}")
            
        except ImportError as e:
            print(f"CommonModelVllm error (vLLM not installed): {e}")
        except Exception as e:
            print(f"CommonModelVllm error: {e}")
        
        # Example 2: Using Qwen3Vllm with thinking mode
        print("\n2. Qwen3Vllm example:")
        try:
            qwen_model = Qwen3Vllm(
                plm="/path/to/qwen/model",
                think_mode=True
            )
            
            responses = qwen_model.batch_generate(
                data=["Solve: 2x + 5 = 13"],
                temperature=0.0
            )
            print(f"Qwen response: {responses}")
            
        except ImportError as e:
            print(f"Qwen3Vllm error (vLLM not installed): {e}")
        except Exception as e:
            print(f"Qwen3Vllm error: {e}")
        
        # Example 3: Using HiragVllm
        print("\n3. HiragVllm example:")
        try:
            hirag_model = HiragVllm(
                plm="/path/to/hirag/model",
                think_mode=True
            )
            
            responses = hirag_model.batch_generate(
                data=["What is the capital of France?"],
                temperature=0.0
            )
            print(f"Hirag response: {responses}")
            
        except ImportError as e:
            print(f"HiragVllm error (vLLM not installed): {e}")
        except Exception as e:
            print(f"HiragVllm error: {e}")
            
    except ImportError as e:
        print(f"Failed to import vLLM models: {e}")

def example_lazy_loading():
    """Example demonstrating lazy loading"""
    print("\n=== Lazy Loading Example ===")
    
    import sys

    # Check what's imported initially
    print("Initially imported modules:")
    api_modules = [m for m in sys.modules.keys() if 'api' in m.lower()]
    vllm_modules = [m for m in sys.modules.keys() if 'vllm' in m.lower()]
    print(f"API-related modules: {api_modules}")
    print(f"vLLM-related modules: {vllm_modules}")
    
    # Import only API models
    print("\nImporting API models...")
    from core.models import APIModel
    print("✓ APIModel imported")
    
    # Check again
    vllm_modules = [m for m in sys.modules.keys() if 'vllm' in m.lower()]
    print(f"vLLM modules after API import: {vllm_modules}")
    
    # Now import vLLM models
    print("\nImporting vLLM models...")
    from core.models import CommonModelVllm
    print("✓ CommonModelVllm imported")
    
    # Check again
    vllm_modules = [m for m in sys.modules.keys() if 'vllm' in m.lower()]
    print(f"vLLM modules after vLLM import: {vllm_modules}")

if __name__ == "__main__":
    print("Models Usage Examples\n")
    print("This script demonstrates how to use the separated models structure.")
    print("Note: Some examples will fail without proper API keys or model paths.\n")
    
    # Run examples
    example_api_usage()
    example_vllm_usage()
    example_lazy_loading()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("Key benefits of this structure:")
    print("1. API and vLLM models are separated")
    print("2. vLLM is only imported when needed (lazy loading)")
    print("3. Missing dependencies are handled gracefully")
    print("4. Backward compatibility is maintained")
    print("5. Clean folder structure with models as a package") 