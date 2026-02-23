# Design Document: Publication Readiness

## Overview

This design document outlines the technical approach for transforming the medical-qwen-distillation project from a functional prototype into a publication-ready, production-quality open-source project. The design focuses on enhancing documentation, testing, code quality, project structure, and developer experience while maintaining backward compatibility with the existing pipeline.

The project currently consists of a four-stage pipeline (data preparation → teacher training → student distillation → evaluation) implemented in Python with PyTorch and HuggingFace Transformers. The publication readiness work will add comprehensive documentation, robust testing infrastructure, improved error handling, Docker support, and automated quality checks without modifying the core pipeline logic.

## Architecture

### High-Level Structure

The enhanced project will maintain the existing pipeline architecture while adding supporting infrastructure:

```
medical-qwen-distillation/
├── src/                          # Core pipeline code (existing)
│   ├── prepare_data.py
│   ├── train_teacher.py
│   ├── distil_student.py
│   ├── evaluate_student.py
│   ├── run_pipeline.py
│   ├── distillation_utils.py
│   └── dataset_adapters.py
├── tests/                        # Enhanced test suite
│   ├── unit/                     # Unit tests for individual modules
│   ├── integration/              # End-to-end pipeline tests
│   ├── property/                 # Property-based tests
│   └── conftest.py               # Shared test fixtures
├── docs/                         # Documentation (new)
│   ├── architecture.md
│   ├── api/                      # Auto-generated API docs
│   ├── guides/                   # User guides
│   └── examples/                 # Usage examples
├── examples/                     # Example scripts and notebooks (new)
│   ├── notebooks/
│   ├── custom_dataset.py
│   ├── inference_only.py
│   └── export_onnx.py
├── configs/                      # Example configurations (new)
│   ├── quick_test.json
│   ├── full_training.json
│   └── multi_dataset.json
├── docker/                       # Docker support (new)
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── .github/                      # Enhanced CI/CD
│   └── workflows/
│       ├── test.yml
│       ├── lint.yml
│       ├── publish.yml
│       └── docker.yml
├── .pre-commit-config.yaml       # Pre-commit hooks (new)
├── CITATION.cff                  # Citation metadata (new)
├── pyproject.toml                # Enhanced with more metadata
└── README.md                     # Comprehensive documentation
```

### Design Principles

1. **Backward Compatibility**: All existing scripts and configurations must continue to work
2. **Minimal Dependencies**: Avoid adding heavy dependencies; use standard library where possible
3. **Gradual Enhancement**: Each component can be improved independently
4. **Developer Experience**: Make it easy for contributors to maintain code quality
5. **Reproducibility**: Ensure consistent results across different environments

## Components and Interfaces

### 1. Enhanced Logging System

**Module**: `src/logging_config.py` (new)

**Purpose**: Provide structured, configurable logging throughout the pipeline

**Interface**:
```python
def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """
    Configure a logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    pass

def log_config(config: dict, logger: logging.Logger) -> None:
    """Log configuration parameters at INFO level."""
    pass

def log_metrics(metrics: dict, logger: logging.Logger, prefix: str = "") -> None:
    """Log metrics dictionary in a formatted way."""
    pass
```

**Integration**: Each pipeline script will import and use this logger instead of print statements.

### 2. Configuration Validation

**Module**: `src/config_validator.py` (new)

**Purpose**: Validate configuration files before pipeline execution

**Interface**:
```python
class ConfigValidator:
    """Validates mcp.json configuration files."""
    
    def __init__(self, config: dict):
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """
        Validate all configuration sections.
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def validate_paths(self) -> None:
        """Validate all path configurations."""
        pass
    
    def validate_model_config(self, model_key: str) -> None:
        """Validate teacher or student model configuration."""
        pass
    
    def validate_training_params(self) -> None:
        """Validate training hyperparameters."""
        pass
    
    def get_report(self) -> str:
        """Return formatted validation report."""
        pass
```

**CLI Integration**: Add `--validate-config` flag to all scripts.

### 3. Enhanced Error Handling

**Module**: `src/exceptions.py` (new)

**Purpose**: Define custom exceptions with helpful error messages

**Interface**:
```python
class MedicalQwenError(Exception):
    """Base exception for all project errors."""
    pass

class ConfigurationError(MedicalQwenError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None):
        self.suggestion = suggestion
        super().__init__(message)

class DatasetError(MedicalQwenError):
    """Raised when dataset loading or processing fails."""
    pass

class ModelError(MedicalQwenError):
    """Raised when model loading or training fails."""
    pass

class InsufficientResourcesError(MedicalQwenError):
    """Raised when GPU memory or disk space is insufficient."""
    
    def __init__(self, resource: str, required: str, available: str):
        self.resource = resource
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient {resource}: required {required}, available {available}"
        )
```

### 4. Type Hints and Docstrings

**Approach**: Add comprehensive type hints and docstrings to all existing modules

**Style**: Google-style docstrings for consistency

**Example Enhancement** (for existing `prepare_data.py`):
```python
def load_config(config_arg: str) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_arg: Path to configuration file (absolute or relative to project root)
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If configuration file does not exist
        json.JSONDecodeError: If configuration file is not valid JSON
        ConfigurationError: If required configuration keys are missing
    """
    pass
```

### 5. Testing Infrastructure

**Structure**:
- `tests/unit/`: Test individual functions and classes in isolation
- `tests/integration/`: Test complete pipeline execution
- `tests/property/`: Property-based tests for data processing
- `tests/conftest.py`: Shared fixtures and test utilities

**Key Test Fixtures**:
```python
@pytest.fixture
def sample_config() -> dict:
    """Provide a minimal valid configuration for testing."""
    pass

@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary directory with sample CSV files."""
    pass

@pytest.fixture
def mock_model():
    """Provide a mock model for testing without GPU."""
    pass
```

**Property-Based Testing**: Use `hypothesis` library for data processing tests

### 6. Documentation Generation

**Tool**: Sphinx with autodoc extension

**Structure**:
```
docs/
├── conf.py                 # Sphinx configuration
├── index.md                # Documentation home
├── quickstart.md           # Getting started guide
├── architecture.md         # System architecture
├── api/                    # Auto-generated API docs
│   ├── prepare_data.md
│   ├── train_teacher.md
│   ├── distil_student.md
│   └── evaluate_student.md
├── guides/
│   ├── custom_datasets.md
│   ├── hyperparameter_tuning.md
│   └── deployment.md
└── examples/
    ├── basic_usage.md
    └── advanced_usage.md
```

**Build Command**: `sphinx-build -b html docs docs/_build`

### 7. Docker Support

**Dockerfiles**:
- `docker/Dockerfile`: CPU-only image (smaller, faster build)
- `docker/Dockerfile.gpu`: CUDA-enabled image for GPU training

**Base Image**: `python:3.10-slim` for CPU, `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` for GPU

**Volume Mounts**:
- `/app/data`: For datasets
- `/app/saved_models`: For model checkpoints
- `/app/mcp.json`: For configuration

**Docker Compose**:
```yaml
version: '3.8'
services:
  medical-qwen:
    build:
      context: .
      dockerfile: docker/Dockerfile
    volumes:
      - ./data:/app/data
      - ./saved_models:/app/saved_models
      - ./mcp.json:/app/mcp.json
    command: python src/run_pipeline.py --config mcp.json
```

### 8. Pre-commit Hooks

**Configuration** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
```

### 9. CI/CD Enhancements

**GitHub Actions Workflows**:

**test.yml**: Run tests on multiple Python versions and OS
```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest, windows-latest]
```

**lint.yml**: Run code quality checks
```yaml
- ruff check
- ruff format --check
- mypy src tests
```

**publish.yml**: Automated PyPI publishing on release tags
```yaml
on:
  release:
    types: [published]
steps:
  - build package
  - publish to PyPI
```

**docker.yml**: Build and push Docker images
```yaml
- build CPU image
- build GPU image
- push to GitHub Container Registry
```

### 10. Example Scripts and Notebooks

**Examples to Create**:

1. **examples/notebooks/quickstart.ipynb**: Complete pipeline walkthrough
2. **examples/custom_dataset.py**: Integrate a custom dataset
3. **examples/inference_only.py**: Use pre-trained models for inference
4. **examples/hyperparameter_search.py**: Grid search for optimal hyperparameters
5. **examples/export_onnx.py**: Export models to ONNX format
6. **examples/evaluate_custom.py**: Evaluate on custom test set

## Data Models

### Configuration Schema

The existing `mcp.json` configuration will be enhanced with validation:

```python
from typing import TypedDict, Optional, List

class PathsConfig(TypedDict):
    data_dir: str
    train_data: str
    validation_data: str
    test_data: str
    output_dir: str
    teacher_model_path: str
    student_model_path: str

class ModelConfig(TypedDict):
    name: str
    max_sequence_length: int
    num_classes: int

class TrainingParams(TypedDict):
    teacher_epochs: int
    teacher_batch_size: int
    teacher_learning_rate: float
    student_epochs: int
    student_batch_size: int
    student_learning_rate: float
    eval_batch_size: int
    logging_steps: int
    weight_decay: Optional[float]

class DistillationParams(TypedDict):
    temperature: float
    alpha: float

class DatasetConfig(TypedDict):
    hf_name: str
    max_samples: Optional[int]

class Config(TypedDict):
    project_name: str
    paths: PathsConfig
    teacher_model: ModelConfig
    student_model: ModelConfig
    training_params: TrainingParams
    distillation_params: DistillationParams
    datasets: List[DatasetConfig]
```

### Test Report Schema

```python
class EvaluationReport(TypedDict):
    accuracy: float
    f1_weighted: float
    classification_report: dict
    timestamp: str
    model_path: str
    test_samples: int
    config_hash: str  # For reproducibility tracking
```

### Logging Record Schema

```python
class PipelineStageLog(TypedDict):
    stage: str  # "prepare_data", "train_teacher", etc.
    start_time: str
    end_time: str
    duration_seconds: float
    status: str  # "success", "failed", "skipped"
    metrics: Optional[dict]
    error: Optional[str]
```


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: API Documentation Completeness

*For any* public function or class in the src/ directory, it should have a docstring that follows Google or NumPy style format.

**Validates: Requirements 1.2, 3.2**

### Property 2: Code Coverage Threshold

*For any* test suite execution, the code coverage percentage should be at least 80% for all modules in src/.

**Validates: Requirements 2.1**

### Property 3: Configuration Validation

*For any* configuration dictionary with invalid values (missing required keys, out-of-range numeric values, non-existent paths), the ConfigValidator should raise a ConfigurationError with a descriptive message.

**Validates: Requirements 2.6, 9.1, 9.5**

### Property 4: Test Suite Performance

*For any* complete test suite execution (excluding integration tests that download models), the total execution time should be under 5 minutes.

**Validates: Requirements 2.4**

### Property 5: Mock-Based Testing

*For any* unit test in the test suite, it should not require GPU availability or download models larger than 10MB.

**Validates: Requirements 2.8**

### Property 6: Type Hint Coverage

*For any* function definition in src/ modules, it should have type hints for all parameters and return values.

**Validates: Requirements 3.1**

### Property 7: Input Validation

*For any* user-facing function that accepts configuration parameters or file paths, it should validate inputs and raise appropriate errors for invalid values.

**Validates: Requirements 3.4, 9.2, 9.6**

### Property 8: Code Formatting Compliance

*For any* Python file in src/ or tests/, running `ruff format --check` should report zero formatting violations.

**Validates: Requirements 3.5**

### Property 9: Linting Compliance

*For any* Python file in src/ or tests/, running `ruff check` should report zero linting warnings or errors.

**Validates: Requirements 3.6**

### Property 10: Function Length Constraint

*For any* function in src/ modules, the number of lines in the function body should not exceed 50 lines.

**Validates: Requirements 3.7**

### Property 11: Logging Usage

*For any* informational output in src/ modules (excluding debug/development code), it should use the logging module rather than print statements.

**Validates: Requirements 3.8**

### Property 12: Test Structure Mirroring

*For any* source file `src/module_name.py`, there should exist a corresponding test file `tests/unit/test_module_name.py` or `tests/integration/test_module_name.py`.

**Validates: Requirements 4.3**

### Property 13: License Header Presence

*For any* Python source file in src/, it should contain a license header comment in the first 10 lines.

**Validates: Requirements 5.6**

### Property 14: Logging Level Configuration

*For any* valid log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL), the logging configuration should accept it and set the logger to that level.

**Validates: Requirements 6.1**

### Property 15: Error Message Quality

*For any* custom exception raised in src/ modules, the error message should be non-empty and contain actionable information.

**Validates: Requirements 9.1, 9.2**

## Error Handling

### Error Categories

1. **Configuration Errors**: Invalid or missing configuration parameters
2. **Data Errors**: Dataset loading, parsing, or validation failures
3. **Model Errors**: Model loading, training, or inference failures
4. **Resource Errors**: Insufficient GPU memory, disk space, or network issues
5. **Validation Errors**: Invalid user inputs or file paths

### Error Handling Strategy

**Fail Fast**: Validate all inputs and configuration before starting expensive operations (model downloads, training)

**Descriptive Messages**: Every error should include:
- What went wrong
- Why it went wrong (if known)
- How to fix it (suggestions)

**Graceful Degradation**: When possible, provide fallback options:
- If GPU is unavailable, fall back to CPU with a warning
- If a dataset fails to load, log the error and continue with other datasets

**Error Recovery**: For transient errors (network issues), implement retry logic with exponential backoff

### Example Error Messages

```python
# Bad
raise ValueError("Invalid config")

# Good
raise ConfigurationError(
    "Invalid teacher_epochs value: -1. Must be a positive integer.",
    suggestion="Set teacher_epochs to a value >= 1 in mcp.json"
)

# Bad
raise FileNotFoundError("File not found")

# Good
raise DatasetError(
    f"Training data file not found: {train_path}",
    suggestion="Run 'python src/prepare_data.py --config mcp.json' to generate data files"
)
```

### Validation Points

1. **Startup Validation**: Validate configuration before any processing
2. **Pre-Training Validation**: Check data files exist and are valid before training
3. **Pre-Distillation Validation**: Verify teacher model exists before distillation
4. **Pre-Evaluation Validation**: Verify student model exists before evaluation

## Testing Strategy

### Dual Testing Approach

The project will use both unit tests and property-based tests for comprehensive coverage:

**Unit Tests**: Verify specific examples, edge cases, and error conditions
- Test specific configuration validation scenarios
- Test error handling for known failure modes
- Test integration between pipeline stages
- Test example scripts execute successfully

**Property Tests**: Verify universal properties across all inputs
- Test data processing functions with random inputs
- Test configuration validation with generated invalid configs
- Test that all public APIs have proper documentation
- Test that code quality standards are maintained

Both testing approaches are complementary and necessary for comprehensive coverage. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across a wide range of inputs.

### Test Organization

```
tests/
├── unit/                           # Unit tests for individual modules
│   ├── test_prepare_data.py
│   ├── test_train_teacher.py
│   ├── test_distil_student.py
│   ├── test_evaluate_student.py
│   ├── test_config_validator.py
│   ├── test_logging_config.py
│   └── test_exceptions.py
├── integration/                    # End-to-end pipeline tests
│   ├── test_full_pipeline.py
│   └── test_quick_pipeline.py
├── property/                       # Property-based tests
│   ├── test_data_processing_properties.py
│   ├── test_config_validation_properties.py
│   └── test_code_quality_properties.py
├── fixtures/                       # Test data and fixtures
│   ├── sample_configs/
│   ├── sample_data/
│   └── mock_models.py
└── conftest.py                     # Shared fixtures
```

### Property-Based Testing Configuration

**Library**: Use `hypothesis` for property-based testing in Python

**Configuration**: Each property test should run a minimum of 100 iterations to ensure comprehensive input coverage

**Tagging**: Each property test must reference its design document property using a comment tag:

```python
# Feature: publication-readiness, Property 3: Configuration Validation
@given(st.dictionaries(st.text(), st.integers()))
def test_config_validation_rejects_invalid_configs(config):
    validator = ConfigValidator(config)
    if not validator.validate():
        assert len(validator.errors) > 0
```

### Test Fixtures

**Shared Fixtures** (in `conftest.py`):

```python
@pytest.fixture
def sample_config() -> dict:
    """Provide a minimal valid configuration for testing."""
    return {
        "project_name": "TestProject",
        "paths": {
            "data_dir": "data/",
            "train_data": "data/train.csv",
            # ... other paths
        },
        # ... other config sections
    }

@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary directory with sample CSV files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample train.csv
    train_df = pd.DataFrame({
        "text": ["sample text 1", "sample text 2"],
        "label": [0, 1]
    })
    train_df.to_csv(data_dir / "train.csv", index=False)
    
    return data_dir

@pytest.fixture
def mock_model():
    """Provide a mock model for testing without GPU."""
    class MockModel:
        def __init__(self):
            self.config = type('Config', (), {'num_labels': 5})()
        
        def __call__(self, *args, **kwargs):
            return type('Output', (), {'logits': torch.randn(1, 5)})()
    
    return MockModel()
```

### CI/CD Testing

**GitHub Actions Workflow** (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests with coverage
        run: |
          pytest tests/ -v --cov=src --cov-report=xml --cov-report=term
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Test Execution

**Local Development**:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run only property tests
pytest tests/property/ -v

# Run specific test file
pytest tests/unit/test_config_validator.py -v
```

**CI Environment**:
- All tests run on every push and pull request
- Coverage reports uploaded to Codecov
- Tests must pass before merging
- Minimum 80% coverage required

### Mock Strategy

To avoid requiring GPU or downloading large models during testing:

1. **Mock HuggingFace Transformers**: Use `unittest.mock` to mock model loading
2. **Mock Dataset Downloads**: Provide small sample datasets in `tests/fixtures/`
3. **Mock CUDA Availability**: Test both CPU and GPU code paths
4. **Mock Network Requests**: Use `responses` library to mock HTTP requests

Example:
```python
@patch('transformers.AutoModelForSequenceClassification.from_pretrained')
def test_train_teacher_without_gpu(mock_model, sample_config, temp_data_dir):
    mock_model.return_value = MockModel()
    # Test training logic without actual model download
    result = train_teacher(sample_config)
    assert result is not None
```

## Implementation Notes

### Backward Compatibility

All enhancements must maintain backward compatibility:
- Existing scripts must continue to work without modification
- Existing `mcp.json` files must remain valid
- New features should be opt-in where possible

### Incremental Rollout

The publication readiness work can be implemented incrementally:

**Phase 1: Foundation**
- Add type hints and docstrings
- Implement logging system
- Add configuration validation
- Create basic test infrastructure

**Phase 2: Quality**
- Add comprehensive unit tests
- Implement pre-commit hooks
- Enhance CI/CD workflows
- Add property-based tests

**Phase 3: Documentation**
- Create documentation directory
- Generate API documentation
- Write user guides
- Create example notebooks

**Phase 4: Distribution**
- Add Docker support
- Enhance PyPI packaging
- Add CITATION.cff
- Create release automation

### Dependencies

**New Dependencies** (to be added to `pyproject.toml`):

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "hypothesis>=6.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "pre-commit>=3.0",
]

docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=1.3",
    "myst-parser>=2.0",
]

examples = [
    "jupyter>=1.0",
    "notebook>=7.0",
    "onnx>=1.14",
    "onnxruntime>=1.15",
]
```

### Performance Considerations

- Logging should have minimal performance impact (use lazy evaluation)
- Validation should be fast (< 1 second for typical configs)
- Tests should run quickly (< 5 minutes total)
- Docker images should be reasonably sized (< 5GB for GPU image)

### Security Considerations

- Validate all file paths to prevent directory traversal
- Sanitize user inputs in configuration
- Use secure defaults (e.g., don't log sensitive information)
- Run security scanning in CI (e.g., `bandit`, `safety`)
