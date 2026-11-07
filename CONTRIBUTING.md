# Contributing to DataType Lab

Thank you for your interest in contributing!

## Development setup

```bash
git clone https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu.git
cd Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Running tests

```bash
pytest -v
```

All tests must pass before opening a pull request.

## Code style

- Follow existing module docstrings (Google style).
- Keep `benchmark_lab/engine.py` free of I/O and Streamlit imports.
- Keep `app.py` free of benchmark logic — delegate everything to `engine.py`.

## Pull request checklist

- [ ] All existing tests pass (`pytest -v`).
- [ ] New behaviour is covered by at least one new test.
- [ ] Docstrings updated for any modified public functions.
