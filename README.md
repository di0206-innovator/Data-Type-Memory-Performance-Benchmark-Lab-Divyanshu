# 🧪 DataType Lab

> **An interactive benchmark laboratory for exploring how NumPy data types affect memory footprint, execution speed, and floating-point precision.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CI](https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu/actions/workflows/tests.yml/badge.svg)](https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu/actions/workflows/tests.yml)
[![Tests](https://img.shields.io/badge/tests-6%20passed-brightgreen)](tests/)

---

DataType Lab is a Streamlit web application that runs live NumPy microbenchmarks directly on your machine. It answers real engineering questions: *How much memory does `int8` save over `int64`? Is `float32` meaningfully faster than `float64` on my hardware? What precision do I lose by switching from `float64` to `float16`?* — all measured in real time, not from hard-coded tables.

Built as a portfolio project to demonstrate clean Python architecture, test-driven development, and data-driven UI design.

---

## ✨ Features

- 📦 **Memory Profiler** — Compare `ndarray.nbytes` footprints across all 12 NumPy dtypes side-by-side with Python `list[int]` and `list[str]` reference rows, with an interactive bar chart.
- ⚡ **Execution Profiler** — Benchmark element-wise add, multiply, compare, and sum with preallocated output buffers and a warm-up pass; median across configurable repeats is the headline figure.
- 🔬 **Precision Lab** — Quantify cast error and summation divergence for `float16`, `float32`, and `float64` against a `float64` ground-truth array.
- 🧭 **Type Advisor** — Enter a `[min, max]` value range and instantly get the smallest standard NumPy integer dtype that fits without overflow.
- 📥 **Export** — Download a combined CSV of all benchmark results or a JSON file capturing full environment metadata (Python version, NumPy build, platform, processor).

---

## 🛠 Tech Stack

| Technology | Role |
|---|---|
| **Python 3.11+** | Core language |
| **NumPy 2.x** | Array creation, dtype introspection, all timed operations |
| **Pandas 2.x** | Result aggregation and CSV serialisation |
| **Plotly 5+** | Interactive bar charts |
| **Streamlit 1.67** | Web UI, session state, download buttons |
| **pytest** | Unit test suite (pure-function engine only) |

---

## 🚀 Quick Start

### macOS / Linux

```bash
git clone https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu.git
cd Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### Windows

```powershell
git clone https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu.git
cd Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
DataType-Lab-GitHub-Project/
├── app.py                      # Streamlit UI entry point (UI only, no benchmark logic)
├── benchmark_lab/
│   ├── __init__.py             # Package metadata (version, author)
│   ├── engine.py               # Pure benchmark functions — no I/O, no Streamlit
│   └── reporting.py            # CSV and JSON export helpers
├── tests/
│   └── test_engine.py          # pytest suite for engine.py (22 tests, class-grouped)
├── pyproject.toml              # Project metadata and pytest configuration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Dev/test dependencies
└── README.md                   # This file
```

---

## 🏗 Architecture

The project enforces a strict three-layer separation:

| Layer | File | Responsibility |
|---|---|---|
| **Engine** | `benchmark_lab/engine.py` | Pure functions only. No I/O, no global state, no Streamlit. Fully testable with pytest. |
| **Reporting** | `benchmark_lab/reporting.py` | Serialises engine output dicts to CSV bytes and JSON bytes for download. |
| **UI** | `app.py` | Streamlit session state, widgets, charts. Calls engine and reporting; does no benchmark logic itself. |

This means `engine.py` and `reporting.py` can be imported and tested in a standard Python environment without a running Streamlit server.

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests with verbose output
pytest -v
```

The test suite covers:

- `make_array` — shape, dtype, determinism, seed independence, error on invalid size, non-negative unsigned values
- `memory_benchmark` — payload byte accuracy, Python list inclusion/exclusion, multi-dtype support
- `speed_benchmark` — 4 operations per dtype, non-negative timings, bool dtype skip, invalid parameter errors
- `precision_benchmark` — 3 float dtypes returned, zero cast error for float64, error ordering, invalid size error
- `recommend_integer_dtype` — small signed/unsigned ranges, full int64 range, out-of-range None, invalid range error, zero-width range
- `system_info` — expected keys present, NumPy version matches runtime

---

## 💡 Design Decisions

- **Preallocated output buffers** — `np.add(a, b, out=buf)` reuses a pre-created buffer so timing measures the BLAS/SIMD kernel cost, not repeated heap allocation. Without this, narrower dtypes would appear artificially slower due to allocation overhead.

- **Median, not mean, for timing** — OS schedulers, background processes, and thermal throttling cause occasional multi-millisecond spikes. The median of 5 repeats is robust to these outliers; the mean would be inflated by even a single spike.

- **Deterministic RNG seeds** — Every `make_array` call uses `numpy.random.default_rng(seed)`. This makes benchmarks reproducible across runs and means test assertions can rely on exact values without mocking.

- **Value ranges safe for all dtypes** — Generated integers are in `[-2, 3)` (signed) or `[0, 5)` (unsigned), and floats in `[-1.0, 1.0)`. This prevents overflow in `int8` add/multiply benchmarks without needing dtype-specific test data.

