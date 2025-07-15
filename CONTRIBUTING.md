# Contributing to PRGB

Thank you for your interest in contributing to PRGB! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- Basic knowledge of RAG systems and evaluation metrics

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/prgb.git
   cd prgb
   ```

3. Install development dependencies:
   ```bash
   make install-dev
   ```

4. Verify the setup:
   ```bash
   make test
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style (see Code Style section)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests and Quality Checks

```bash
make test          # Run all tests
make lint          # Check code style
make format        # Format code
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new evaluation metric"
```

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Code

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Keep functions and classes focused and well-documented
- Use meaningful variable and function names

### Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstring format
- Update README.md for user-facing changes
- Add inline comments for complex logic

### Testing

- Write unit tests for new functionality
- Aim for at least 80% code coverage
- Use descriptive test names
- Test both success and failure cases

## Project Structure

```
PRGB/
├── README.md
├── pyproject.toml
├── requirements.txt
├── eval.py
├── test_imports.py
├── run_eval.sh
├── Makefile
├── CONTRIBUTING.md
├── CHANGELOG.md
├── .gitignore
├── .flake8
├── .pre-commit-config.yaml
├── LEGAL.md
│
├── core/
│   ├── __init__.py
│   ├── eval.py
│   ├── eval_apis.py
│   ├── eval_draft.py
│   ├── data.py
│   ├── models.py
│   └── eval_types.py
│
├── config/
│   ├── api_prompt_config_ch.json
│   ├── api_prompt_config_en.json
│   └── default_prompt_config.json
│
├── utils/
│   ├── __init__.py
│   ├── filter_mutual_right_samples.py
│   ├── filter_similar_query.py
│   └── transfer_csv_to_jsonl.py
│
├── examples/
│   └── basic_evaluation.py
│
└── tests/
    ├── test_data_process.py
    ├── test_checkanswer.py
    └── test.jsonl
```

## Adding New Features

### New Evaluation Metrics

1. Add the metric implementation in `core/metrics.py`
2. Add corresponding tests in `tests/test_metrics.py`
3. Update the evaluation pipeline in `core/eval.py`
4. Update documentation

### New Model Support

1. Add model class in `core/models.py`
2. Implement required methods (generate, batch_generate)
3. Add tests in `tests/test_models.py`
4. Update configuration files if needed

### New Data Formats

1. Add data loader in `core/data.py`
2. Add format validation
3. Add tests for the new format
4. Update documentation

## Reporting Issues

When reporting issues, please include:

1. **Environment**: OS, Python version, package versions
2. **Steps to reproduce**: Clear, step-by-step instructions
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Error messages**: Full error traceback if applicable
6. **Additional context**: Any other relevant information

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No new warnings are introduced
- [ ] Code is self-documenting and well-commented

### PR Description

Include:
- Summary of changes
- Motivation for the change
- Testing performed
- Any breaking changes
- Screenshots (if UI changes)

## Code Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Review**: At least one maintainer reviews the PR
3. **Feedback**: Address any review comments
4. **Merge**: PR is merged once approved

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Release Steps

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and publish to PyPI

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Documentation**: Check the README and inline documentation

## Code of Conduct

Please be respectful and inclusive in all interactions. We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/).

## License

By contributing to PRGB, you agree that your contributions will be licensed under the same license as the project.

## Thank You

Thank you for contributing to PRGB! 🚀 