"""DataType Lab — Streamlit application entry point.

Responsibility: UI only. All benchmark logic lives in ``benchmark_lab.engine``
and all export logic in ``benchmark_lab.reporting`` so this module stays thin
and testable independently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from benchmark_lab.engine import (
    FLOAT_DTYPES,
    MEMORY_DTYPES,
    NUMERIC_DTYPES,
    memory_benchmark,
    precision_benchmark,
    recommend_integer_dtype,
    speed_benchmark,
    system_info,
)
from benchmark_lab.reporting import metadata_json, results_bundle_csv

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DataType Lab | Memory & Performance Benchmarks",
    page_icon="🧪",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu",
        "Report a bug": "https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu/issues",
        "About": "DataType Lab — Interactive NumPy dtype benchmark laboratory.",
    },
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for _key in ("memory_results", "speed_results", "precision_results"):
    if _key not in st.session_state:
        st.session_state[_key] = []

# ---------------------------------------------------------------------------
# Sidebar — experiment configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("Adjust parameters then run each benchmark independently.")

    dataset_size = st.slider(
        "Dataset size (elements)",
        min_value=1_000,
        max_value=2_000_000,
        value=100_000,
        step=1_000,
        help="Number of array elements. Memory and speed tests use this value directly.",
    )
    chosen_memory = st.multiselect(
        "Memory test dtypes",
        options=list(MEMORY_DTYPES),
        default=["bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64"],
        help="Select which dtypes to include in the memory footprint comparison.",
    )
    chosen_speed = st.multiselect(
        "Speed test dtypes",
        options=list(NUMERIC_DTYPES),
        default=["int8", "int16", "int32", "int64", "float16", "float32", "float64"],
        help="Select which numeric dtypes to include in the execution time benchmark.",
    )

    st.markdown("---")
    st.markdown("### 🕐 Timing settings")
    repeats = st.slider(
        "Timing repeats",
        min_value=3,
        max_value=9,
        value=5,
        help="Number of independent timing samples. More repeats → more stable median.",
    )
    iterations = st.slider(
        "Iterations per repeat",
        min_value=1,
        max_value=10,
        value=3,
        help="Calls per sample; elapsed time is divided by this count.",
    )

    st.markdown("---")
    with st.expander("ℹ️ How to use"):
        st.markdown(
            """
            1. Set **dataset size** and choose **dtypes** above.
            2. Open a tab and click its **Run** button.
            3. Each tab stores results independently — run any combination.
            4. Use **Export** at the bottom to download CSV or JSON.
            """
        )

    st.markdown("---")
    st.caption("🧪 DataType Lab · [GitHub](https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu)")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown("# 🧪 DataType Lab")
st.markdown(
    "**An interactive benchmark laboratory** for exploring how NumPy data types affect "
    "memory footprint, execution speed, and floating-point precision — "
    "measured live on this machine, not hard-coded."
)

# ---------------------------------------------------------------------------
# Environment metrics strip
# ---------------------------------------------------------------------------

info = system_info()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset size", f"{dataset_size:,}")
c2.metric("Python", info["python"])
c3.metric("NumPy", info["numpy"])
c4.metric("Platform", info["platform"].split("-")[0])

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

memory_tab, speed_tab, precision_tab, advisor_tab, methodology_tab = st.tabs([
    "📦 Memory profiler",
    "⚡ Execution profiler",
    "🔬 Precision lab",
    "🧭 Type advisor",
    "📖 Methodology",
])

# ── Memory ─────────────────────────────────────────────────────────────────

with memory_tab:
    st.subheader("Memory footprint comparison")
    st.markdown(
        "Compares the raw data-buffer size (`ndarray.nbytes`) of NumPy arrays against "
        "shallow-size estimates of Python `list[int]` and `list[str]` for context. "
        "These are **different memory models** — NumPy uses a contiguous typed buffer; "
        "Python lists store pointers to individual heap-allocated objects."
    )

    if st.button("▶ Run memory benchmark", type="primary", disabled=not chosen_memory, key="run_memory"):
        with st.spinner("Measuring memory footprints…"):
            st.session_state.memory_results = memory_benchmark(dataset_size, chosen_memory)

    if st.session_state.memory_results:
        memory_df = pd.DataFrame(st.session_state.memory_results)
        display_df = memory_df.copy()
        display_df["payload_MiB"] = (display_df["payload_bytes"] / (1024 * 1024)).round(4)
        display_df["bytes_per_element"] = display_df["bytes_per_element"].round(2)
        st.dataframe(
            display_df[[
                "representation", "elements", "payload_bytes",
                "payload_MiB", "bytes_per_element", "measurement",
            ]],
            use_container_width=True,
            hide_index=True,
        )
        fig = px.bar(
            display_df,
            x="representation",
            y="payload_MiB",
            color="representation",
            title="Storage footprint by representation",
            labels={"representation": "Representation", "payload_MiB": "Memory (MiB)"},
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        smallest = memory_df.sort_values("bytes_per_element").iloc[0]
        st.info(
            f"**Most compact in this run:** {smallest['representation']} "
            f"at **{smallest['bytes_per_element']:.2f} bytes/element**."
        )
    else:
        st.info("Select at least one dtype and click **▶ Run memory benchmark**.")

# ── Speed ───────────────────────────────────────────────────────────────────

with speed_tab:
    st.subheader("Execution time comparison")
    st.markdown(
        "Benchmarks element-wise **add**, **multiply**, **compare**, and **sum** for each "
        "selected dtype. Output buffers are preallocated before timing, so results measure "
        "the operation cost only. One warm-up call is excluded. The **median** across repeats "
        "is reported as the primary statistic."
    )

    if st.button("▶ Run execution benchmark", type="primary", disabled=not chosen_speed, key="run_speed"):
        with st.spinner("Running timed operations — larger datasets take longer…"):
            st.session_state.speed_results = speed_benchmark(
                dataset_size, chosen_speed, repeats=repeats, iterations=iterations
            )

    if st.session_state.speed_results:
        speed_df = pd.DataFrame(st.session_state.speed_results)
        st.dataframe(
            speed_df[["dtype", "operation", "elements", "median_ms", "min_ms", "std_ms", "ns_per_element"]].round(4),
            use_container_width=True,
            hide_index=True,
        )
        selected_operation = st.selectbox(
            "Chart operation",
            options=list(speed_df["operation"].drop_duplicates()),
            key="speed_operation_chart",
        )
        chart_df = speed_df[speed_df["operation"] == selected_operation]
        fig = px.bar(
            chart_df,
            x="dtype",
            y="median_ms",
            color="dtype",
            error_y="std_ms",
            title=f"Median execution time — {selected_operation}",
            labels={"median_ms": "Median time (ms)", "dtype": "NumPy dtype"},
            template="plotly_white",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        fastest = chart_df.sort_values("median_ms").iloc[0]
        st.info(
            f"**Fastest for this operation:** `{fastest['dtype']}` "
            f"at **{fastest['median_ms']:.4f} ms** median. "
            "Results are hardware- and workload-dependent."
        )
    else:
        st.info("Select numeric dtypes and click **▶ Run execution benchmark**.")

# ── Precision ───────────────────────────────────────────────────────────────

with precision_tab:
    st.subheader("Floating-point precision comparison")
    st.markdown(
        "Values are generated in `float64` and cast to each floating-point type. "
        "**Cast error** measures the representational loss. **Sum error** shows "
        "how accumulation diverges from a `float64` reference sum. "
        "The sample is capped at 10,000 elements to keep `float16` summation within range."
    )

    if st.button("▶ Run precision benchmark", type="primary", key="run_precision"):
        with st.spinner("Comparing floating-point precision…"):
            st.session_state.precision_results = precision_benchmark(dataset_size)

    if st.session_state.precision_results:
        precision_df = pd.DataFrame(st.session_state.precision_results)
        st.dataframe(precision_df.round(10), use_container_width=True, hide_index=True)
        fig = px.bar(
            precision_df,
            x="dtype",
            y="mean_abs_cast_error",
            color="dtype",
            title="Mean absolute representation error by dtype",
            labels={"mean_abs_cast_error": "Mean absolute error", "dtype": "Floating-point dtype"},
            template="plotly_white",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click **▶ Run precision benchmark** to compare `float16`, `float32`, and `float64`.")

# ── Advisor ─────────────────────────────────────────────────────────────────

with advisor_tab:
    st.subheader("Integer data type advisor")
    st.markdown(
        "Enter the minimum and maximum values your data will ever hold. "
        "The advisor returns the **smallest standard NumPy integer dtype** "
        "that can represent the entire range without overflow."
    )
    c1, c2 = st.columns(2)
    minimum = c1.number_input("Minimum value", value=-100, step=1, format="%d")
    maximum = c2.number_input("Maximum value", value=100, step=1, format="%d")

    if minimum <= maximum:
        recommendation = recommend_integer_dtype(int(minimum), int(maximum))
        if recommendation:
            itemsize = np.dtype(recommendation).itemsize
            st.success(
                f"✅ Recommended dtype: **`{recommendation}`** "
                f"({itemsize} byte{'s' if itemsize != 1 else ''} per element)."
            )
        else:
            st.warning("⚠️ No standard NumPy integer dtype can represent this range.")
    else:
        st.error("❌ Minimum value must be ≤ maximum value.")

    st.caption(
        "This advisor checks range safety only. "
        "Choose a wider type if future values may exceed the current bounds."
    )

# ── Methodology ─────────────────────────────────────────────────────────────

with methodology_tab:
    st.subheader("Methodology & interpretation")
    st.markdown(
        """
        | Topic | Detail |
        |---|---|
        | **Memory measurement** | `ndarray.nbytes` = data buffer only. Container/object overhead shown separately and must not be added to payload bytes. |
        | **Python list estimates** | `sys.getsizeof(list)` + sum of `sys.getsizeof` per element. Shallow only — does not recursively traverse all referenced objects. |
        | **Timing** | Arrays pre-allocated before timing. 1 warm-up excluded. Median of *repeats* samples; each sample averaged over *iterations* calls. |
        | **Arithmetic safety** | Generated values are small (integers in [-2, 3), floats in [-1.0, 1.0)) to prevent fixed-width overflow during speed tests. |
        | **Precision** | `float64` reference treated as ground truth. Cast error and sum error shown. Not a full numerical-stability study. |
        | **Environment** | CPU, OS, power state, background load, NumPy BLAS build, and memory bandwidth all affect results. Re-run under consistent conditions. |
        | **Scope** | NumPy microbenchmark — not a universal dtype ranking. A faster dtype for one operation may be slower for another. |
        """
    )
    with st.expander("🖥️ Full environment details"):
        st.json(info)

# ---------------------------------------------------------------------------
# Export section
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📥 Export results")

all_memory = st.session_state.memory_results
all_speed = st.session_state.speed_results
all_precision = st.session_state.precision_results

if any((all_memory, all_speed, all_precision)):
    col_csv, col_json = st.columns(2)
    with col_csv:
        st.download_button(
            label="⬇ Download combined results (CSV)",
            data=results_bundle_csv(all_memory, all_speed, all_precision),
            file_name="datatype_lab_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_json:
        st.download_button(
            label="⬇ Download environment metadata (JSON)",
            data=metadata_json({
                "dataset_size": dataset_size,
                "memory_dtypes": chosen_memory,
                "speed_dtypes": chosen_speed,
                "timing_repeats": repeats,
                "iterations_per_repeat": iterations,
                "python_list_limit": 100_000,
            }),
            file_name="benchmark_environment.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    st.caption("Run at least one experiment to enable CSV and environment exports.")

st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem;margin-top:2rem'>"
    "🧪 DataType Lab · Results generated locally · "
    "<a href='https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
