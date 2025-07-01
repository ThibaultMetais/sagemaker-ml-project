# Test Suite Documentation

This directory contains comprehensive unit tests for the entire `src/` directory of the SageMaker project. The test suite is designed to provide excellent coverage, proper mocking, and impeccable documentation.

## Test Structure

### Test Files

- `conftest.py` - Pytest configuration and shared fixtures
- `test_config.py` - Tests for configuration management (`src/config.py`)
- `test_utils.py` - Tests for utility functions (`src/utils.py`)
- `test_prepare_data.py` - Tests for data preparation (`src/prepare_data.py`)
- `test_training.py` - Tests for model training (`src/training.py`)
- `test_estimator.py` - Tests for SageMaker estimator creation (`src/estimator.py`)
- `test_hyperparameter_tuning.py` - Tests for hyperparameter tuning (`src/hyperparameter_tuning.py`)
- `test_deploy.py` - Tests for model deployment (`src/deploy.py`)
- `test_mlflow_inference.py` - Tests for MLflow inference (`src/test_mlflow_inference.py`)

### Test Coverage

The test suite provides comprehensive coverage for:

1. **Configuration Management** - Environment variables, AWS services, and configuration classes
2. **Utility Functions** - Requirements generation, hash checking, and build script execution
3. **Data Preparation** - UCI dataset fetching, preprocessing, visualization, and S3 upload
4. **Model Training** - MLflow integration, model evaluation, and argument parsing
5. **SageMaker Integration** - Estimator creation, hyperparameter configuration, and build script integration
6. **Hyperparameter Tuning** - Tuner creation, hyperparameter ranges, and optimization settings
7. **Model Deployment** - ECR operations, Docker container building, and SageMaker deployment
8. **MLflow Inference** - Model loading from runs and registry, and prediction execution

## Testing Strategy

### Mocking Strategy

The test suite uses extensive mocking to ensure:

- **No External Dependencies** - All AWS services, MLflow, Docker, and external APIs are mocked
- **Isolated Tests** - Each test runs independently without side effects
- **Fast Execution** - No network calls or heavy operations during testing
- **Predictable Results** - Controlled test environment with known inputs and outputs

### Key Fixtures

The `conftest.py` file provides shared fixtures for:

- `test_data` - Sample test data for training and testing
- `mock_env_vars` - Mocked environment variables
- `mock_aws_services` - Mocked AWS services (STS, S3, ECR)
- `mock_mlflow` - Mocked MLflow tracking and model registry
- `temp_files` - Temporary files for testing file operations
- `mock_subprocess` - Mocked subprocess calls
- `mock_docker` - Mocked Docker operations
- `mock_sagemaker_session` - Mocked SageMaker session
- `mock_estimator` - Mocked SageMaker Estimator
- `mock_hyperparameter_tuner` - Mocked SageMaker HyperparameterTuner
- `mock_ucimlrepo` - Mocked UCI dataset fetching
- `mock_matplotlib` - Mocked matplotlib operations
- `mock_boto3_client` - Mocked boto3 client creation

### Test Categories

Each test file contains tests for:

1. **Happy Path** - Normal operation with expected inputs
2. **Edge Cases** - Boundary conditions and unusual inputs
3. **Error Handling** - Exception scenarios and error recovery
4. **Integration** - Component interaction and data flow
5. **Configuration** - Different configuration scenarios
6. **Validation** - Input validation and parameter checking

## Running the Tests

### Prerequisites

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

### Basic Test Execution

Run all tests:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

### Coverage Reports

Run tests with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Generate HTML coverage report:

```bash
pytest --cov=src --cov-report=html
```

The HTML report will be generated in `htmlcov/index.html`.

### Running Specific Test Files

Run tests for a specific module:

```bash
pytest tests/test_config.py
pytest tests/test_training.py
```

### Running Specific Test Classes

Run tests for a specific class:

```bash
pytest tests/test_config.py::TestConfig
pytest tests/test_training.py::TestTrainFunction
```

### Running Specific Test Methods

Run a specific test:

```bash
pytest tests/test_config.py::TestConfig::test_aws_account_id_property
```

### Parallel Test Execution

Run tests in parallel (requires pytest-xdist):

```bash
pip install pytest-xdist
pytest -n auto
```

## Test Quality Standards

### Documentation

Every test includes:

- **Descriptive Names** - Test names clearly describe what is being tested
- **Docstrings** - Detailed explanations of test purpose and expected behavior
- **Comments** - Inline comments explaining complex test logic

### Assertions

Tests use meaningful assertions:

- **Specific Assertions** - Use specific assertion methods (e.g., `assert_called_once_with` instead of `assert_called`)
- **Clear Messages** - Assertion messages explain what went wrong
- **Multiple Checks** - Test multiple aspects of the same operation

### Mock Verification

Tests verify that mocks are called correctly:

- **Call Counts** - Verify functions are called the expected number of times
- **Call Arguments** - Verify functions are called with correct parameters
- **Call Order** - Verify functions are called in the expected order when relevant

### Error Testing

Tests include error scenarios:

- **Exception Testing** - Test that appropriate exceptions are raised
- **Error Recovery** - Test error handling and recovery mechanisms
- **Edge Cases** - Test boundary conditions and invalid inputs

## Continuous Integration

The test suite is designed to run in CI/CD pipelines:

- **No External Dependencies** - Tests don't require AWS credentials or external services
- **Fast Execution** - Tests complete quickly for rapid feedback
- **Deterministic** - Tests produce consistent results
- **Isolated** - Tests don't interfere with each other

## Coverage Goals

The test suite aims for:

- **Line Coverage** - >95% of source code lines executed
- **Branch Coverage** - >90% of conditional branches tested
- **Function Coverage** - 100% of functions called
- **Edge Case Coverage** - All error paths and boundary conditions tested

## Best Practices

### Writing New Tests

When adding new tests:

1. **Follow Naming Convention** - Use descriptive test names
2. **Use Existing Fixtures** - Leverage shared fixtures when possible
3. **Mock External Dependencies** - Never make real network calls
4. **Test Both Success and Failure** - Include positive and negative test cases
5. **Document Complex Logic** - Add comments for non-obvious test logic

### Maintaining Tests

When maintaining tests:

1. **Keep Tests Focused** - Each test should verify one specific behavior
2. **Avoid Test Interdependence** - Tests should not depend on each other
3. **Update Mocks** - Keep mocks in sync with actual function signatures
4. **Review Coverage** - Ensure new code is adequately tested

### Debugging Tests

When debugging test failures:

1. **Check Mock Setup** - Verify mocks are configured correctly
2. **Review Test Data** - Ensure test data matches expected format
3. **Check Assertions** - Verify assertions match actual behavior
4. **Use Debug Output** - Add print statements or use pytest -s for debugging

## Performance Considerations

The test suite is optimized for:

- **Fast Execution** - Tests complete in under 30 seconds
- **Low Memory Usage** - Tests don't create large objects unnecessarily
- **Efficient Mocking** - Mocks are lightweight and fast
- **Parallel Execution** - Tests can run in parallel when using pytest-xdist

## Future Enhancements

Potential improvements to the test suite:

1. **Property-Based Testing** - Use hypothesis for property-based testing
2. **Performance Testing** - Add performance benchmarks
3. **Integration Testing** - Add limited integration tests with real services
4. **Visual Regression Testing** - Test matplotlib/seaborn output
5. **Load Testing** - Test with larger datasets and higher loads 