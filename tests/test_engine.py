"""Unit tests for benchmark_lab.engine.

Each test is intentionally scoped to a single behaviour so failures are
easy to diagnose.  Small array sizes are used to keep the test suite fast;
correctness does not depend on scale.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmark_lab.engine import (
    FLOAT_DTYPES,
    MEMORY_DTYPES,
    NUMERIC_DTYPES,
    make_array,
    memory_benchmark,
    precision_benchmark,
    recommend_integer_dtype,
    speed_benchmark,
    system_info,
)


# ---------------------------------------------------------------------------
# make_array
# ---------------------------------------------------------------------------

class TestMakeArray:
    def test_shape_and_dtype(self):
        arr = make_array(128, "int16")
        assert arr.shape == (128,)
        assert arr.dtype == np.dtype("int16")

    def test_bool_dtype(self):
        arr = make_array(64, "bool")
        assert arr.dtype == np.bool_
        assert set(arr.tolist()).issubset({True, False})

    def test_deterministic_seed(self):
        a = make_array(50, "float32", seed=99)
        b = make_array(50, "float32", seed=99)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = make_array(50, "float32", seed=1)
        b = make_array(50, "float32", seed=2)
        assert not np.array_equal(a, b)

    def test_size_zero_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            make_array(0, "int32")

    def test_unsigned_non_negative(self):
        arr = make_array(1000, "uint8")
        assert arr.min() >= 0


# ---------------------------------------------------------------------------
# memory_benchmark
# ---------------------------------------------------------------------------

class TestMemoryBenchmark:
    def test_numpy_payload_bytes(self):
        rows = memory_benchmark(100, ["int32"], include_python_list=False)
        assert len(rows) == 1
        assert rows[0]["payload_bytes"] == 400   # 100 elements x 4 bytes
        assert rows[0]["bytes_per_element"] == 4.0

    def test_includes_python_lists_by_default(self):
        rows = memory_benchmark(50, ["float32"])
        labels = {r["representation"] for r in rows}
        assert "Python list[int]" in labels
        assert "Python list[str]" in labels

    def test_python_list_excluded_when_flag_false(self):
        rows = memory_benchmark(50, ["float32"], include_python_list=False)
        assert all("Python" not in r["representation"] for r in rows)

    def test_multiple_dtypes(self):
        rows = memory_benchmark(10, ["int8", "float64"], include_python_list=False)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# speed_benchmark
# ---------------------------------------------------------------------------

class TestSpeedBenchmark:
    def test_returns_four_operations_per_dtype(self):
        rows = speed_benchmark(64, ["int8"], repeats=3, iterations=1)
        assert len(rows) == 4

    def test_all_timings_non_negative(self):
        rows = speed_benchmark(64, ["float32"], repeats=3, iterations=1)
        assert all(r["median_ms"] >= 0 for r in rows)
        assert all(r["min_ms"] >= 0 for r in rows)

    def test_bool_dtype_skipped(self):
        """Boolean arrays are not numeric; speed_benchmark should skip them."""
        rows = speed_benchmark(64, ["bool"], repeats=3, iterations=1)
        assert rows == []

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            speed_benchmark(0, ["int32"])

    def test_invalid_repeats_raises(self):
        with pytest.raises(ValueError, match="Repeats"):
            speed_benchmark(10, ["int32"], repeats=0)

    def test_invalid_iterations_raises(self):
        with pytest.raises(ValueError, match="Iterations"):
            speed_benchmark(10, ["int32"], iterations=0)


# ---------------------------------------------------------------------------
# precision_benchmark
# ---------------------------------------------------------------------------

class TestPrecisionBenchmark:
    def test_returns_three_float_dtypes(self):
        rows = precision_benchmark(100)
        assert [r["dtype"] for r in rows] == list(FLOAT_DTYPES)

    def test_float64_has_zero_cast_error(self):
        rows = precision_benchmark(100)
        float64_row = next(r for r in rows if r["dtype"] == "float64")
        assert float64_row["max_abs_cast_error"] == pytest.approx(0.0)

    def test_float16_has_larger_error_than_float32(self):
        rows = precision_benchmark(1000)
        by_dtype = {r["dtype"]: r for r in rows}
        assert by_dtype["float16"]["mean_abs_cast_error"] > by_dtype["float32"]["mean_abs_cast_error"]

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            precision_benchmark(0)


# ---------------------------------------------------------------------------
# recommend_integer_dtype
# ---------------------------------------------------------------------------

class TestRecommendIntegerDtype:
    def test_small_signed_range(self):
        assert recommend_integer_dtype(-10, 10) == "int8"

    def test_small_unsigned_range(self):
        assert recommend_integer_dtype(0, 200) == "uint8"

    def test_full_int64_range(self):
        assert recommend_integer_dtype(-(2**63), 2**63 - 1) == "int64"

    def test_out_of_range_returns_none(self):
        assert recommend_integer_dtype(-(2**64), 2**64) is None

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            recommend_integer_dtype(10, -10)

    def test_zero_range(self):
        """A single value range should return the smallest fitting type."""
        result = recommend_integer_dtype(0, 0)
        assert result in ("int8", "uint8")


# ---------------------------------------------------------------------------
# system_info
# ---------------------------------------------------------------------------

class TestSystemInfo:
    def test_returns_expected_keys(self):
        info = system_info()
        assert set(info.keys()) == {"python", "numpy", "platform", "processor"}

    def test_numpy_version_matches(self):
        info = system_info()
        assert info["numpy"] == np.__version__
