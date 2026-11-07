"""Core benchmark routines for DataType Lab.

Design principles:
- **Pure functions**: no I/O, no global state, no Streamlit imports.
- **Separation of concerns**: array creation is decoupled from timing so
  measurements reflect the operation cost alone.
- **Deterministic seeds**: every array is reproducible; benchmarks can be
  compared across runs given the same configuration.
- **Preallocated output buffers**: add, multiply, and compare reuse output
  arrays so timings exclude repeated buffer allocation.
- **Median over mean**: the median is the primary timing statistic because
  it is robust to occasional OS jitter spikes that inflate the mean.
"""
from __future__ import annotations

import platform
import statistics
import sys
import time
from typing import Callable, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Public dtype registries – used by the UI to populate multiselect widgets.
# ---------------------------------------------------------------------------

NUMERIC_DTYPES: tuple[str, ...] = (
    "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64",
)
"""All numeric dtypes available for speed benchmarks."""

MEMORY_DTYPES: tuple[str, ...] = ("bool",) + NUMERIC_DTYPES
"""All dtypes available for memory benchmarks (includes bool)."""

FLOAT_DTYPES: tuple[str, ...] = ("float16", "float32", "float64")
"""Floating-point dtypes used in the precision benchmark."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dtype(name: str) -> np.dtype:
    """Validate and return a NumPy dtype, raising ValueError for bad inputs.

    Args:
        name: A NumPy dtype string (e.g. ``"int32"``, ``"float64"``).

    Returns:
        The corresponding :class:`numpy.dtype` object.

    Raises:
        ValueError: If *name* is not a recognised NumPy dtype string, or if
            the dtype kind is not Boolean, integer, or floating-point.
    """
    try:
        dtype = np.dtype(name)
    except (TypeError, KeyError) as exc:
        raise ValueError(f"Unsupported dtype: {name!r}") from exc
    if dtype.kind not in "biuf":
        raise ValueError(
            f"Only Boolean, integer and floating dtypes are supported: {name!r}"
        )
    return dtype


def _time_callable(
    func: Callable[[], object], repeats: int, iterations: int
) -> tuple[float, float, float]:
    """Time a callable and return robust statistics in milliseconds per call.

    One warm-up invocation is performed before recording begins so that
    instruction-cache and branch-predictor effects do not skew the first
    recorded sample.

    Args:
        func: Zero-argument callable to benchmark.
        repeats: Number of independent timing samples to collect.
        iterations: Number of ``func`` calls per sample; the elapsed time is
            divided by *iterations* to obtain a per-call figure.

    Returns:
        A three-tuple ``(median_ms, min_ms, pstdev_ms)`` where each value is
        a float in milliseconds per call.
    """
    func()  # warm-up — excluded from recorded samples
    samples_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            func()
        elapsed_ns = time.perf_counter_ns() - start
        # Divide by iterations to get the average cost of a single call.
        samples_ms.append((elapsed_ns / iterations) / 1_000_000)
    return (
        float(statistics.median(samples_ms)),
        float(min(samples_ms)),
        float(statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_array(size: int, dtype_name: str, seed: int = 42) -> np.ndarray:
    """Create a deterministic array of low-magnitude values safe for arithmetic.

    Values are intentionally small so that element-wise add and multiply do
    not overflow even the narrowest fixed-width integer types (e.g. int8).

    Args:
        size: Number of elements. Must be >= 1.
        dtype_name: NumPy dtype string accepted by :func:`_dtype`.
        seed: Integer seed for :class:`numpy.random.Generator`. The default
            value ``42`` produces the same array on every call.

    Returns:
        A 1-D NumPy array of *size* elements with the requested dtype.

    Raises:
        ValueError: If *size* < 1 or *dtype_name* is not supported.
    """
    if size < 1:
        raise ValueError("Dataset size must be at least 1.")
    dtype = _dtype(dtype_name)
    rng = np.random.default_rng(seed)

    if dtype.kind == "b":
        values = rng.integers(0, 2, size=size, dtype=np.uint8).astype(np.bool_)
    elif dtype.kind == "u":
        # Range [0, 5) keeps values well inside every unsigned integer type.
        values = rng.integers(0, 5, size=size, dtype=np.uint8).astype(dtype)
    elif dtype.kind == "i":
        # Range [-2, 3) keeps values inside int8 and all wider signed types.
        values = rng.integers(-2, 3, size=size, dtype=np.int8).astype(dtype)
    else:
        values = rng.uniform(-1.0, 1.0, size=size).astype(dtype)

    return values


def memory_benchmark(
    size: int,
    dtype_names: Iterable[str],
    include_python_list: bool = True,
    python_list_limit: int = 100_000,
) -> list[dict[str, object]]:
    """Measure raw memory usage for NumPy arrays and Python list representations.

    NumPy payload is measured with ``ndarray.nbytes`` (data buffer only).
    Python list figures are shallow-size estimates: list container bytes plus
    the sum of ``sys.getsizeof`` for each element object.  These are *not*
    byte-for-byte equivalent representations and should not be compared
    directly with NumPy's contiguous buffer.

    Args:
        size: Number of elements per array.
        dtype_names: Iterable of NumPy dtype strings to benchmark.
        include_python_list: When *True*, append ``list[int]`` and
            ``list[str]`` reference rows for context.
        python_list_limit: Maximum element count for Python list rows; capped
            to keep the UI responsive for large *size* values.

    Returns:
        A list of dicts, each describing one representation with keys:
        ``representation``, ``dtype``, ``elements``, ``payload_bytes``,
        ``container_bytes``, ``bytes_per_element``, ``measurement``.
    """
    rows: list[dict[str, object]] = []

    for name in dtype_names:
        arr = make_array(size, name)
        rows.append({
            "representation": f"NumPy {name}",
            "dtype": name,
            "elements": size,
            "payload_bytes": int(arr.nbytes),
            "container_bytes": int(sys.getsizeof(arr)),
            "bytes_per_element": float(arr.nbytes / size),
            "measurement": "ndarray payload (nbytes); container estimate shown separately",
        })

    if include_python_list:
        # --- list[int] ---
        list_size = min(size, python_list_limit)
        int_values = list(range(10_000, 10_000 + list_size))
        list_bytes = sys.getsizeof(int_values) + sum(
            sys.getsizeof(v) for v in int_values
        )
        rows.append({
            "representation": "Python list[int]",
            "dtype": "Python int objects",
            "elements": list_size,
            "payload_bytes": int(list_bytes),
            "container_bytes": int(sys.getsizeof(int_values)),
            "bytes_per_element": float(list_bytes / list_size),
            "measurement": "list container + each integer object's shallow size",
        })

        # --- list[str] ---
        string_size = min(size, python_list_limit)
        str_values = [f"item_{i:06d}" for i in range(string_size)]
        string_bytes = sys.getsizeof(str_values) + sum(
            sys.getsizeof(v) for v in str_values
        )
        rows.append({
            "representation": "Python list[str]",
            "dtype": "Python str objects",
            "elements": string_size,
            "payload_bytes": int(string_bytes),
            "container_bytes": int(sys.getsizeof(str_values)),
            "bytes_per_element": float(string_bytes / string_size),
            "measurement": "list container + each string object's shallow size",
        })

    return rows


def speed_benchmark(
    size: int,
    dtype_names: Iterable[str],
    repeats: int = 5,
    iterations: int = 3,
) -> list[dict[str, object]]:
    """Benchmark element-wise add, multiply, compare, and sum for numeric dtypes.

    Input arrays and preallocated output buffers are created before timing
    begins.  Add, multiply, and compare reuse those output buffers so their
    timings measure the operation cost only, not buffer allocation.

    Sum uses a ``float64`` accumulator for a consistent policy across all
    input dtypes, making the results directly comparable.

    Args:
        size: Number of elements. Must be >= 1.
        dtype_names: Iterable of NumPy dtype strings to benchmark.
        repeats: Number of independent timing samples. Must be >= 1.
        iterations: Calls per sample; median is taken. Must be >= 1.

    Returns:
        A list of dicts with keys: ``dtype``, ``operation``, ``elements``,
        ``median_ms``, ``min_ms``, ``std_ms``, ``ns_per_element``.

    Raises:
        ValueError: If *size*, *repeats*, or *iterations* are out of range.
    """
    if size < 1:
        raise ValueError("Dataset size must be at least 1.")
    if repeats < 1:
        raise ValueError("Repeats must be at least 1.")
    if iterations < 1:
        raise ValueError("Iterations must be at least 1.")

    rows: list[dict[str, object]] = []
    for name in dtype_names:
        dtype = _dtype(name)
        if dtype.kind not in "iuf":
            # Boolean arrays are not meaningful for arithmetic benchmarks.
            continue

        # Arrays and output buffers are created outside the timed region.
        a = make_array(size, name, seed=101)
        b = make_array(size, name, seed=202)
        add_out = np.empty_like(a)
        multiply_out = np.empty_like(a)
        compare_out = np.empty(size, dtype=np.bool_)

        operations: tuple[tuple[str, Callable[[], object]], ...] = (
            ("add (preallocated output)",      lambda: np.add(a, b, out=add_out)),
            ("multiply (preallocated output)", lambda: np.multiply(a, b, out=multiply_out)),
            ("compare (preallocated output)",  lambda: np.greater(a, b, out=compare_out)),
            ("sum (float64 accumulator)",      lambda: np.sum(a, dtype=np.float64)),
        )

        for operation, func in operations:
            median_ms, min_ms, std_ms = _time_callable(func, repeats, iterations)
            rows.append({
                "dtype": name,
                "operation": operation,
                "elements": size,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "std_ms": std_ms,
                # ns_per_element gives a size-independent throughput figure.
                "ns_per_element": median_ms * 1_000_000 / size,
            })

    return rows


def precision_benchmark(size: int, seed: int = 777) -> list[dict[str, object]]:
    """Compare cast error and summation accuracy for float16, float32, float64.

    A float64 reference array is cast to each floating-point dtype.  The
    absolute element-wise difference measures representational error.  The
    dtype-specific ``np.sum`` result is compared with a float64 reference
    sum to show accumulation error.

    The sample is capped at 10,000 elements so float16 summation stays within
    a safe numeric range (float16 max approx 65,504; 10,000 x 1.0 = 10,000 < max).

    Args:
        size: Requested element count; capped at 10,000 internally.
        seed: RNG seed for reproducibility.

    Returns:
        A list of three dicts (one per float dtype) with keys:
        ``dtype``, ``sample_elements``, ``mean_abs_cast_error``,
        ``max_abs_cast_error``, ``sum_result``, ``reference_sum_float64``,
        ``absolute_sum_error``, ``bytes_per_element``.

    Raises:
        ValueError: If *size* < 1.
    """
    if size < 1:
        raise ValueError("Dataset size must be at least 1.")

    # Cap to prevent float16 overflow during summation.
    sample_size = min(size, 10_000)
    rng = np.random.default_rng(seed)
    reference = rng.uniform(0.0, 1.0, size=sample_size).astype(np.float64)
    reference_sum = float(np.sum(reference, dtype=np.float64))

    rows: list[dict[str, object]] = []
    for name in FLOAT_DTYPES:
        typed = reference.astype(np.dtype(name))
        restored = typed.astype(np.float64)
        absolute_error = np.abs(restored - reference)
        typed_sum = float(np.sum(typed, dtype=np.dtype(name)))
        rows.append({
            "dtype": name,
            "sample_elements": sample_size,
            "mean_abs_cast_error": float(np.mean(absolute_error)),
            "max_abs_cast_error": float(np.max(absolute_error)),
            "sum_result": typed_sum,
            "reference_sum_float64": reference_sum,
            "absolute_sum_error": float(abs(typed_sum - reference_sum)),
            "bytes_per_element": int(np.dtype(name).itemsize),
        })

    return rows


def system_info() -> dict[str, str]:
    """Return environment metadata for attaching to benchmark exports.

    Returns:
        A dict with keys ``python``, ``numpy``, ``platform``, ``processor``.
    """
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine() or "Not reported by Python",
    }


def recommend_integer_dtype(min_value: int, max_value: int) -> str | None:
    """Return the smallest standard NumPy integer dtype that fits a value range.

    Checks both signed and unsigned candidates in ascending size order and
    returns the first one whose representable range fully covers
    [*min_value*, *max_value*].

    Args:
        min_value: Inclusive lower bound of the value range.
        max_value: Inclusive upper bound of the value range.

    Returns:
        The dtype name string (e.g. ``"int8"``) or ``None`` if no standard
        NumPy integer dtype can represent the range.

    Raises:
        ValueError: If *min_value* > *max_value*.
    """
    if min_value > max_value:
        raise ValueError(
            f"min_value ({min_value}) must be <= max_value ({max_value})."
        )
    candidates = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
    fitting: list[tuple[int, str]] = []
    for name in candidates:
        info = np.iinfo(np.dtype(name))
        if info.min <= min_value and max_value <= info.max:
            fitting.append((np.dtype(name).itemsize, name))
    return min(fitting)[1] if fitting else None
