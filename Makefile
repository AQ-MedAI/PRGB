.PHONY: help install install-dev test lint format clean docs test-imports

# Default values for evaluation parameters
MODEL_NAME ?= Qwen3
MODEL_NAME ?= Qwen3_infer
BATCH_SIZE ?= 16
DATA_PATH_ZH ?= data/zh.jsonl
DATA_PATH_EN ?= data/en.jsonl
OUTPUT_PATH ?= ./results

# Default target
help:
	@echo "PRGB - RAG System Evaluation Tool"
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install the package and dependencies"
	@echo "  install-dev  Install the package with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-imports Test if imports work correctly"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  eval         Run evaluation (requires EVAL_MODEL_PATH env var)"
	@echo "  eval-ch      Run Chinese evaluation (data/zh.jsonl)"
	@echo "  eval-ch-infer Run Chinese evaluation in inference mode (data/zh.jsonl)"
	@echo "  eval-en      Run English evaluation (data/en.jsonl)"
	@echo "  eval-en-infer Run English evaluation in inference mode (data/en.jsonl)"
	@echo "  eval-test    Run evaluation with test data"
	@echo "  export-errors Export error samples (requires EVAL_RESULT_FILE env var)"
	@echo ""
	@echo "Usage examples:"
	@echo "  export EVAL_MODEL_PATH=/path/to/your/model && make eval"
	@echo "  EVAL_MODEL_PATH=/path/to/your/model make eval"
	@echo "  EVAL_MODEL_PATH=/path/to/your/model make eval-ch"
	@echo "  EVAL_MODEL_PATH=/path/to/your/model MODEL_NAME=Qwen2.5 make eval-ch"
	@echo "  EVAL_MODEL_PATH=/path/to/your/model make eval-ch-infer"
	@echo "  EVAL_MODEL_PATH=/path/to/your/model make eval-en"
	@echo "  EVAL_MODEL_PATH=/path/to/your/model make eval-en-infer"
	@echo "  EVAL_RESULT_FILE=results/model_eval_result.jsonl make export-errors"
	@echo ""
	@echo "Default values:"
	@echo "  MODEL_NAME=$(MODEL_NAME)"
	@echo "  MODEL_NAME=$(MODEL_NAME)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  DATA_PATH_ZH=$(DATA_PATH_ZH)"
	@echo "  DATA_PATH_EN=$(DATA_PATH_EN)"
	@echo "  OUTPUT_PATH=$(OUTPUT_PATH)"
	@echo ""
	@echo "You can override defaults:"
	@echo "  MODEL_NAME=Qwen2.5 make eval-ch"
	@echo "  BATCH_SIZE=8 make eval-ch"
	@echo "  EVAL_MODEL_PATH=/path/to/model MODEL_NAME=Gemma3 BATCH_SIZE=32 make eval-ch"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=core --cov=utils --cov-report=html

test-quick:
	pytest tests/ -v

test-imports:
	python test_imports.py

# Code quality
lint:
	flake8 core/ utils/ tests/ eval.py
	isort --check-only core/ utils/ tests/ eval.py

format:
	black core/ utils/ tests/ eval.py
	isort core/ utils/ tests/ eval.py

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	cd docs && make html

# Example evaluation (requires GPU)
# Usage: make eval (requires EVAL_MODEL_PATH environment variable)
eval:
	@echo "Running example evaluation..."
	@echo "Note: This requires GPU and model files"
	@if [ -z "$(EVAL_MODEL_PATH)" ]; then \
		echo "Error: EVAL_MODEL_PATH environment variable is required."; \
		echo "Please set it first: export EVAL_MODEL_PATH=/path/to/your/model"; \
		echo "Or run: EVAL_MODEL_PATH=/path/to/your/model make eval"; \
		exit 1; \
	fi
	@echo "Using model path: $(EVAL_MODEL_PATH)"
	@echo "Using model name: $(MODEL_NAME)"
	@echo "Using batch size: $(BATCH_SIZE)"
	python eval.py \
		--model-name "$(MODEL_NAME)" \
		--model-path "$(EVAL_MODEL_PATH)" \
		--data-path "$(DATA_PATH_ZH)" \
		--output-path "$(OUTPUT_PATH)" \
		--batch-size $(BATCH_SIZE)
	python eval.py \
		--model-name "$(MODEL_NAME)" \
		--model-path "$(EVAL_MODEL_PATH)" \
		--data-path "$(DATA_PATH_EN)" \
		--output-path "$(OUTPUT_PATH)" \
		--batch-size $(BATCH_SIZE)

# Chinese evaluation (requires GPU)
# Usage: make eval-ch (requires EVAL_MODEL_PATH environment variable)
eval-ch:
	@echo "Running Chinese evaluation..."
	@echo "Note: This requires GPU and model files"
	@if [ -z "$(EVAL_MODEL_PATH)" ]; then \
		echo "Error: EVAL_MODEL_PATH environment variable is required."; \
		echo "Please set it first: export EVAL_MODEL_PATH=/path/to/your/model"; \
		echo "Or run: EVAL_MODEL_PATH=/path/to/your/model make eval-ch"; \
		exit 1; \
	fi
	@echo "Using model path: $(EVAL_MODEL_PATH)"
	@echo "Using model name: $(MODEL_NAME)"
	@echo "Using Chinese data: $(DATA_PATH_ZH)"
	@echo "Using batch size: $(BATCH_SIZE)"
	python eval.py \
		--model-name "$(MODEL_NAME)" \
		--model-path "$(EVAL_MODEL_PATH)" \
		--data-path "$(DATA_PATH_ZH)" \
		--output-path "$(OUTPUT_PATH)" \
		--batch-size $(BATCH_SIZE)

# Chinese evaluation in inference mode (requires GPU)
# Usage: make eval-ch-infer (requires EVAL_MODEL_PATH environment variable)
eval-ch-infer:
	@echo "Running Chinese evaluation in inference mode..."
	@echo "Note: This requires GPU and model files"
	@if [ -z "$(EVAL_MODEL_PATH)" ]; then \
		echo "Error: EVAL_MODEL_PATH environment variable is required."; \
		echo "Please set it first: export EVAL_MODEL_PATH=/path/to/your/model"; \
		echo "Or run: EVAL_MODEL_PATH=/path/to/your/model make eval-ch-infer"; \
		exit 1; \
	fi
	@echo "Using model path: $(EVAL_MODEL_PATH)"
	@echo "Using model name: $(MODEL_NAME)"
	@echo "Using Chinese data: $(DATA_PATH_ZH)"
	@echo "Using batch size: $(BATCH_SIZE)"
	@echo "Running in inference mode with $(MODEL_NAME) model"
	python eval.py \
		--model-name "$(MODEL_NAME)" \
		--model-path "$(EVAL_MODEL_PATH)" \
		--inference-mode true \
		--data-path "$(DATA_PATH_ZH)" \
		--output-path "$(OUTPUT_PATH)" \
		--batch-size $(BATCH_SIZE)

# English evaluation (requires GPU)
# Usage: make eval-en (requires EVAL_MODEL_PATH environment variable)
eval-en:
	@echo "Running English evaluation..."
	@echo "Note: This requires GPU and model files"
	@if [ -z "$(EVAL_MODEL_PATH)" ]; then \
		echo "Error: EVAL_MODEL_PATH environment variable is required."; \
		echo "Please set it first: export EVAL_MODEL_PATH=/path/to/your/model"; \
		echo "Or run: EVAL_MODEL_PATH=/path/to/your/model make eval-en"; \
		exit 1; \
	fi
	@echo "Using model path: $(EVAL_MODEL_PATH)"
	@echo "Using model name: $(MODEL_NAME)"
	@echo "Using English data: $(DATA_PATH_EN)"
	@echo "Using batch size: $(BATCH_SIZE)"
	python eval.py \
		--model-name "$(MODEL_NAME)" \
		--model-path "$(EVAL_MODEL_PATH)" \
		--data-path "$(DATA_PATH_EN)" \
		--output-path "$(OUTPUT_PATH)" \
		--batch-size $(BATCH_SIZE)

# English evaluation in inference mode (requires GPU)
# Usage: make eval-en-infer (requires EVAL_MODEL_PATH environment variable)
eval-en-infer:
	@echo "Running English evaluation in inference mode..."
	@echo "Note: This requires GPU and model files"
	@if [ -z "$(EVAL_MODEL_PATH)" ]; then \
		echo "Error: EVAL_MODEL_PATH environment variable is required."; \
		echo "Please set it first: export EVAL_MODEL_PATH=/path/to/your/model"; \
		echo "Or run: EVAL_MODEL_PATH=/path/to/your/model make eval-en-infer"; \
		exit 1; \
	fi
	@echo "Using model path: $(EVAL_MODEL_PATH)"
	@echo "Using model name: $(MODEL_NAME)"
	@echo "Using English data: $(DATA_PATH_EN)"
	@echo "Using batch size: $(BATCH_SIZE)"
	@echo "Running in inference mode with $(MODEL_NAME) model"
	python eval.py \
		--model-name "$(MODEL_NAME)" \
		--model-path "$(EVAL_MODEL_PATH)" \
		--inference-mode true \
		--data-path "$(DATA_PATH_EN)" \
		--output-path "$(OUTPUT_PATH)" \
		--batch-size $(BATCH_SIZE)

# Quick evaluation with test data
eval-test:
	@echo "Running evaluation with test data..."
	@echo "Using model name: test"
	@echo "Using batch size: 1"
	python eval.py \
		--model-name "test" \
		--model-path "test" \
		--data-path "tests/test.jsonl" \
		--output-path "$(OUTPUT_PATH)" \
		--batch-size 1 \
		--temperature 0.7

# Export error samples from evaluation results
# Usage: make export-errors EVAL_RESULT_FILE=results/model_eval_result.jsonl
export-errors:
	@echo "Exporting error samples..."
	@if [ -z "$(EVAL_RESULT_FILE)" ]; then \
		echo "Error: EVAL_RESULT_FILE environment variable is required."; \
		echo "Please set it first: export EVAL_RESULT_FILE=results/model_eval_result.jsonl"; \
		echo "Or run: EVAL_RESULT_FILE=results/model_eval_result.jsonl make export-errors"; \
		exit 1; \
	fi
	@echo "Using result file: $(EVAL_RESULT_FILE)"
	python examples/export_errors.py

# Development setup
setup-dev: install-dev test-imports
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

# Check project structure
check-structure:
	@echo "Checking project structure..."
	@test -f README.md || echo "Missing: README.md"
	@test -f pyproject.toml || echo "Missing: pyproject.toml"
	@test -f requirements.txt || echo "Missing: requirements.txt"
	@test -d core || echo "Missing: core/ directory"
	@test -d utils || echo "Missing: utils/ directory"
	@test -d config || echo "Missing: config/ directory"
	@test -d tests || echo "Missing: tests/ directory"
	@echo "Structure check complete!"