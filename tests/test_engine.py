import numpy as np
import pytest

from benchmark_lab.engine import (
    make_array,
    memory_benchmark,
    precision_benchmark,
    recommend_integer_dtype,
    speed_benchmark,
)


def test_make_array_has_requested_size_and_dtype():
    arr = make_array(128, "int16")
    assert arr.shape == (128,)
    assert arr.dtype == np.dtype("int16")


def test_memory_benchmark_reports_numpy_payload():
    rows = memory_benchmark(100, ["int32"], include_python_list=False)
    assert len(rows) == 1
    assert rows[0]["payload_bytes"] == 400
    assert rows[0]["bytes_per_element"] == 4


def test_speed_benchmark_returns_all_operations():
    rows = speed_benchmark(64, ["int8"], repeats=3, iterations=1)
    assert len(rows) == 4
    assert all(row["median_ms"] >= 0 for row in rows)


def test_precision_benchmark_contains_three_float_formats():
    rows = precision_benchmark(100)
    assert [row["dtype"] for row in rows] == ["float16", "float32", "float64"]
    assert rows[-1]["max_abs_cast_error"] == pytest.approx(0.0)


def test_integer_dtype_advisor():
    assert recommend_integer_dtype(-10, 10) == "int8"
    assert recommend_integer_dtype(0, 200) == "uint8"
    assert recommend_integer_dtype(-(2**63), 2**63 - 1) == "int64"
    assert recommend_integer_dtype(-(2**64), 2**64) is None


def test_invalid_range_raises():
    with pytest.raises(ValueError):
        recommend_integer_dtype(10, -10)
