# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-15

### Added

- Initial release of PRGB
- RAG system evaluation framework
- Support for multiple evaluation metrics
- Batch processing capabilities
- Configuration-driven evaluation
- Multi-language support (Chinese and English)
- Noise injection for robustness testing
- Detailed result reporting

### Features

- **Model Support**: Qwen3, CommonModelVllm, InferModelVllm
- **Evaluation Metrics**: Accuracy, Exact Match, F1 Score
- **Data Formats**: JSONL support with extensible format system
- **Configuration**: JSON configuration files
- **Logging**: Comprehensive logging with configurable levels
- **Testing**: Unit tests and integration tests

### Technical Details

- Python 3.7+ compatibility
- GPU acceleration support via VLLM
- Modular architecture for easy extension
- Type hints throughout the codebase
- Comprehensive error handling

---

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
