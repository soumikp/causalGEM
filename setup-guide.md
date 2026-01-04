# Complete Setup Guide: From Code to PyPI

This guide walks you through creating your GitHub repository and publishing causalGEM to PyPI.

## Step 1: Create the Repository Structure

```bash
# Create directory structure
mkdir -p causalGEM/{src/causalgem,tests,examples,docs,.github/workflows}
cd causalGEM

# Create empty files
touch src/causalgem/__init__.py
touch src/causalgem/py.typed  # For type hints
touch tests/__init__.py
```

## Step 2: Copy the Package Files

Copy these files from the artifacts:

| Artifact | Destination |
|----------|-------------|
| `pyproject.toml` | `./pyproject.toml` |
| `README.md` | `./README.md` |
| `src/causalgem/__init__.py` | `./src/causalgem/__init__.py` |
| `src/causalgem/estimator.py` | `./src/causalgem/estimator.py` |
| `src/causalgem/diagnostics.py` | `./src/causalgem/diagnostics.py` |
| `src/causalgem/utils.py` | `./src/causalgem/utils.py` |
| `src/causalgem/simulation.py` | `./src/causalgem/simulation.py` |
| `tests/test_estimator.py` | `./tests/test_estimator.py` |
| `examples/benchmark_tuebingen.py` | `./examples/benchmark_tuebingen.py` |
| `.github/workflows/publish.yml` | `./.github/workflows/publish.yml` |

## Step 3: Create Additional Files

### LICENSE (MIT)

```text
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### .gitignore

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/
*.egg

# Virtual environments
venv/
.venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage.xml

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

### MANIFEST.in

```text
include LICENSE
include README.md
include pyproject.toml
recursive-include src *.py *.typed
recursive-include tests *.py
recursive-include examples *.py
```

## Step 4: Update Package Metadata

Edit `pyproject.toml`:
1. Update `authors` with your information
2. Update `project.urls` with your GitHub repository URL

Edit `src/causalgem/__init__.py`:
1. Update `__author__`
2. Add your paper citation in the docstring

## Step 5: Test Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Test the package
python -c "from causalgem import estimate_causal_direction; print('Success!')"
```

## Step 6: Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: causalGEM - Generative Exposure Model for causal discovery"
```

## Step 7: Create GitHub Repository

1. Go to https://github.com/new
2. Name: `causalGEM`
3. Description: "Bivariate causal discovery via the Generative Exposure Model (GEM)"
4. Make it public
5. Don't initialize with README (we have one)

```bash
git remote add origin https://github.com/yourusername/causalGEM.git
git branch -M main
git push -u origin main
```

## Step 8: Set Up PyPI Publishing

### Option A: Trusted Publishing (Recommended)

1. **Create PyPI Account**: https://pypi.org/account/register/
2. **Create TestPyPI Account**: https://test.pypi.org/account/register/

3. **Configure Trusted Publishers on PyPI**:
   - Go to https://pypi.org/manage/account/publishing/
   - Add new pending publisher:
     - PyPI Project Name: `causalGEM`
     - Owner: `yourusername`
     - Repository name: `causalGEM`
     - Workflow name: `publish.yml`
     - Environment name: `pypi`

4. **Configure Trusted Publishers on TestPyPI**:
   - Go to https://test.pypi.org/manage/account/publishing/
   - Same settings but environment: `testpypi`

5. **Create GitHub Environments**:
   - Go to your repo → Settings → Environments
   - Create `pypi` environment
   - Create `testpypi` environment

## Step 9: Publish Your Package

### Test Release (TestPyPI)

```bash
# Create a test tag
git tag v0.1.0-test
git push origin v0.1.0-test
```

This triggers the TestPyPI workflow. Check: https://test.pypi.org/project/causalGEM/

### Production Release

1. Go to GitHub → Releases → Create new release
2. Tag: `v0.1.0`
3. Title: `v0.1.0 - Initial Release`
4. Description: Release notes
5. Publish release

This triggers the PyPI workflow. Your package will be at: https://pypi.org/project/causalGEM/

## Step 10: Install and Verify

```bash
# From PyPI
pip install causalGEM

# Test it works
python -c "from causalgem import estimate_causal_direction; print('Installed successfully!')"
```

## Quick Usage Example

```python
import numpy as np
from causalgem import estimate_causal_direction

# Generate example data: X causes Y
np.random.seed(42)
x = np.random.uniform(0, 1, 500)
y = x**2 + np.random.normal(0, 0.1, 500)

# Estimate causal direction
result = estimate_causal_direction(x, y)
print(result)
# Output: CausalGEMResult(C_{X→Y} = ..., decision = 'X → Y', ...)
```

## Updating the Package

For future updates:

1. Update version in `pyproject.toml` and `__init__.py`
2. Commit changes
3. Create new GitHub release with new tag (e.g., `v0.2.0`)
4. Package automatically publishes to PyPI

## Troubleshooting

### Import Errors
- Ensure `src/causalgem/__init__.py` exists
- Check `[tool.setuptools.packages.find]` in `pyproject.toml`

### Test Failures
- Run `pip install -e ".[dev]"` to get test dependencies
- Check Python version compatibility

### PyPI Upload Errors
- Verify package name isn't taken (check https://pypi.org/project/causalGEM/)
- Check trusted publisher configuration
- Ensure tag format matches workflow triggers

## Files Summary

```
causalGEM/
├── README.md                    # Package documentation
├── LICENSE                      # MIT license
├── pyproject.toml              # Package configuration
├── MANIFEST.in                 # Package manifest
├── .gitignore                  # Git ignore rules
├── .github/
│   └── workflows/
│       └── publish.yml         # CI/CD for PyPI
├── src/
│   └── causalgem/
│       ├── __init__.py         # Package init & exports
│       ├── py.typed            # Type hints marker
│       ├── estimator.py        # Core GEM estimator
│       ├── diagnostics.py      # Assumption diagnostics
│       ├── utils.py            # Utility functions
│       └── simulation.py       # Data generation
├── tests/
│   ├── __init__.py
│   └── test_estimator.py       # Unit tests
└── examples/
    └── benchmark_tuebingen.py  # Benchmark example
```
