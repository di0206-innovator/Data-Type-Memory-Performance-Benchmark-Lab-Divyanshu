# DataType Lab — Data Type Memory Performance Benchmark Laboratory

A Python + Streamlit application for comparing memory footprint, execution time, and floating-point precision across common NumPy data types.

## Features

- Memory profiling for NumPy Boolean, signed/unsigned integer, and floating-point arrays.
- Separate footprint estimates for Python `list[int]` and `list[str]` representations.
- Timed add, multiply, comparison, and reduction operations.
- Float16 / float32 / float64 representation and summation-error analysis.
- Integer dtype advisor that finds the smallest standard NumPy integer type able to represent a requested value range.
- Interactive Plotly charts and CSV / JSON result exports.
- Environment metadata and experiment settings included in JSON exports.

## Requirements

- Python 3.10 or newer recommended.
- pip and an internet connection for the first dependency installation.

## Run locally

### macOS / Linux

```bash
cd datatype-memory-benchmark-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
cd datatype-memory-benchmark-lab
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in a browser.

## Run tests

```bash
pip install pytest
pytest -q
```

## Suggested demonstration sequence

1. Open **Memory profiler**, keep the default data types, and run the benchmark.
2. Compare the NumPy payload bytes. For a fixed element count, a dtype's raw array storage is `number_of_elements × dtype.itemsize`.
3. Open **Execution profiler** and run the speed tests. Change the dataset size and repeat to observe workload scaling.
4. Open **Precision lab** and compare float16, float32 and float64 errors.
5. Use **Type advisor** with a sample integer range such as -100 to 100.
6. Export the CSV and environment JSON for the project report.

## Methodology and interpretation notes

- `ndarray.nbytes` is the size of an array's element buffer. The ndarray object's reported size is shown separately because it includes array-object overhead and may include the owned data buffer; do not add these figures together as if they were independent memory allocations.
- Python list totals are approximate shallow-size estimates: list container plus each element object's `sys.getsizeof` value. They do not recursively account for every possible referenced allocation and should not be presented as a byte-for-byte equivalent of NumPy's contiguous buffer.
- Speed-test input arrays and output buffers are allocated before timing. Add/multiply/comparison re-use output buffers. Reduction uses a float64 accumulator for a consistent policy. One warm-up is excluded and median time across repeated batches is reported.
- The generated integer values are deliberately small to avoid overflow during arithmetic. The tests are microbenchmarks; results depend on hardware, NumPy build, OS, power state, memory bandwidth, and background load.
- The precision experiment uses a deterministic float64 reference and up to 10,000 values. It reports cast error and dtype-specific sum error; it is not a full numerical-stability study.

## Project structure

```text
app.py                     Streamlit user interface
benchmark_lab/engine.py    Benchmark and dtype recommendation functions
benchmark_lab/reporting.py CSV and environment export helpers
tests/test_engine.py       Basic automated tests
PROJECT_REPORT.md          Report draft and result-entry template
requirements.txt           Python dependencies
```

## Limitations / future improvements

- Add process-level peak RAM measurement using `psutil` in a separate isolated subprocess to reduce measurement noise.
- Add more workloads such as sorting, matrix multiplication, serialization, and file I/O.
- Add benchmark warm-up controls, CPU affinity, and repeatability checks for controlled lab environments.
- Add optional hardware-counter profiling where supported.
