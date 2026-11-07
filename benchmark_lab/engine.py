"""Core benchmark routines for the Data Type Memory Performance Benchmark Lab.

The functions here deliberately separate array creation from timed operations so
that execution measurements focus on the operation rather than data generation.
"""
from __future__ import annotations

import platform
import statistics
import sys
import time
from typing import Callable, Iterable

import numpy as np

NUMERIC_DTYPES: tuple[str, ...] = (
    "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64",
)
MEMORY_DTYPES: tuple[str, ...] = ("bool",) + NUMERIC_DTYPES
FLOAT_DTYPES: tuple[str, ...] = ("float16", "float32", "float64")


def _dtype(name: str) -> np.dtype:
    try:
        dtype = np.dtype(name)
    except (TypeError, KeyError) as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc
    if dtype.kind not in "biuf":
        raise ValueError(f"Only Boolean, integer and floating dtypes are supported: {name}")
    return dtype


def make_array(size: int, dtype_name: str, seed: int = 42) -> np.ndarray:
    """Create deterministic, low-magnitude data safe for fixed-width arithmetic."""
    if size < 1:
        raise ValueError("Dataset size must be at least 1.")
    dtype = _dtype(dtype_name)
    rng = np.random.default_rng(seed)
    if dtype.kind == "b":
        values = rng.integers(0, 2, size=size, dtype=np.uint8).astype(np.bool_)
    elif dtype.kind == "u":
        # Small non-negative values prevent overflow in the add/multiply tests.
        values = rng.integers(0, 5, size=size, dtype=np.uint8).astype(dtype)
    elif dtype.kind == "i":
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
    """Measure NumPy payload bytes and Python object/list footprint estimates."""
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
        list_size = min(size, python_list_limit)
        values = list(range(10_000, 10_000 + list_size))
        list_bytes = sys.getsizeof(values) + sum(sys.getsizeof(value) for value in values)
        rows.append({
            "representation": "Python list[int]",
            "dtype": "Python int objects",
            "elements": list_size,
            "payload_bytes": int(list_bytes),
            "container_bytes": int(sys.getsizeof(values)),
            "bytes_per_element": float(list_bytes / list_size),
            "measurement": "list container + each integer object's shallow size",
        })

    # Include a realistic variable-length string representation as a separate,
    # non-NumPy comparison. The sample is capped to keep the UI responsive.
    if include_python_list:
        string_size = min(size, python_list_limit)
        strings = [f"item_{i:06d}" for i in range(string_size)]
        string_bytes = sys.getsizeof(strings) + sum(sys.getsizeof(value) for value in strings)
        rows.append({
            "representation": "Python list[str]",
            "dtype": "Python str objects",
            "elements": string_size,
            "payload_bytes": int(string_bytes),
            "container_bytes": int(sys.getsizeof(strings)),
            "bytes_per_element": float(string_bytes / string_size),
            "measurement": "list container + each string object's shallow size",
        })
    return rows


def _time_callable(
    func: Callable[[], object], repeats: int, iterations: int
) -> tuple[float, float, float]:
    """Return median, minimum and population standard deviation in ms/op."""
    func()  # warm-up outside recorded samples
    samples_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            func()
        elapsed_ns = time.perf_counter_ns() - start
        samples_ms.append((elapsed_ns / iterations) / 1_000_000)
    return (
        float(statistics.median(samples_ms)),
        float(min(samples_ms)),
        float(statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0),
    )


def speed_benchmark(
    size: int,
    dtype_names: Iterable[str],
    repeats: int = 5,
    iterations: int = 3,
) -> list[dict[str, object]]:
    """Benchmark add, multiply, comparison and sum for numeric NumPy dtypes.

    Add/multiply/comparison use preallocated output arrays, excluding repeated
    output allocation from those timings. Sum uses a float64 accumulator for a
    consistent accumulation policy across input dtypes.
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
            continue
        a = make_array(size, name, seed=101)
        b = make_array(size, name, seed=202)
        add_out = np.empty_like(a)
        multiply_out = np.empty_like(a)
        compare_out = np.empty(size, dtype=np.bool_)
        operations: tuple[tuple[str, Callable[[], object]], ...] = (
            ("add (preallocated output)", lambda: np.add(a, b, out=add_out)),
            ("multiply (preallocated output)", lambda: np.multiply(a, b, out=multiply_out)),
            ("compare (preallocated output)", lambda: np.greater(a, b, out=compare_out)),
            ("sum (float64 accumulator)", lambda: np.sum(a, dtype=np.float64)),
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
                "ns_per_element": median_ms * 1_000_000 / size,
            })
    return rows


def precision_benchmark(size: int, seed: int = 777) -> list[dict[str, object]]:
    """Compare float casts and dtype-specific sum results against float64.

    A bounded 10,000-element sample avoids half-precision sum overflow while
    still showing representational and accumulation differences.
    """
    if size < 1:
        raise ValueError("Dataset size must be at least 1.")
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
    """Return environment metadata to attach to benchmark exports."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine() or "Not reported by Python",
    }


def recommend_integer_dtype(min_value: int, max_value: int) -> str | None:
    """Return the smallest standard NumPy integer dtype that safely fits a range."""
    if min_value > max_value:
        raise ValueError("Minimum value must be less than or equal to maximum value.")
    candidates = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
    fitting = []
    for name in candidates:
        info = np.iinfo(np.dtype(name))
        if info.min <= min_value and max_value <= info.max:
            fitting.append((np.dtype(name).itemsize, name))
    return min(fitting)[1] if fitting else None
