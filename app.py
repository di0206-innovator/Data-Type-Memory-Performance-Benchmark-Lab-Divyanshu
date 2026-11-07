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

st.set_page_config(
    page_title="DataType Lab | Memory & Performance Benchmarks",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 DataType Lab")
st.caption("Data Type Memory Performance Benchmark Laboratory")
st.write(
    "Explore how numeric and Python data representations affect memory footprint, "
    "execution time, and numerical precision. Results are measured on this machine; "
    "they are not hard-coded examples."
)

with st.sidebar:
    st.header("Experiment configuration")
    dataset_size = st.slider("Dataset size (elements)", 1_000, 2_000_000, 100_000, 1_000)
    chosen_memory = st.multiselect(
        "Data types for memory tests",
        options=list(MEMORY_DTYPES),
        default=["bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64"],
    )
    chosen_speed = st.multiselect(
        "Data types for speed tests",
        options=list(NUMERIC_DTYPES),
        default=["int8", "int16", "int32", "int64", "float16", "float32", "float64"],
    )
    repeats = st.slider("Timing repeats", min_value=3, max_value=9, value=5)
    iterations = st.slider("Iterations per repeat", min_value=1, max_value=10, value=3)
    st.caption("Memory and speed tests use the selected dataset size. Precision uses up to 10,000 values.")

# State is kept in the session so the user can run each experiment independently.
for key in ("memory_results", "speed_results", "precision_results"):
    if key not in st.session_state:
        st.session_state[key] = []

info = system_info()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset size", f"{dataset_size:,}")
col2.metric("Python", info["python"])
col3.metric("NumPy", info["numpy"])
col4.metric("Platform", info["platform"].split("-")[0])

memory_tab, speed_tab, precision_tab, advisor_tab, methodology_tab = st.tabs(
    ["Memory profiler", "Execution profiler", "Precision lab", "Type advisor", "Methodology"]
)

with memory_tab:
    st.subheader("Memory footprint comparison")
    st.write(
        "NumPy `nbytes` reports the array's data buffer. The ndarray container size is "
        "shown separately. Python list rows include the list container and the shallow "
        "size of each contained Python object; these are different representations, "
        "not apples-to-apples fixed-width numeric arrays."
    )
    if st.button("Run memory benchmark", type="primary", disabled=not chosen_memory, key="run_memory"):
        with st.spinner("Measuring memory footprints..."):
            st.session_state.memory_results = memory_benchmark(dataset_size, chosen_memory)
    if st.session_state.memory_results:
        memory_df = pd.DataFrame(st.session_state.memory_results)
        display_df = memory_df.copy()
        display_df["payload_MiB"] = display_df["payload_bytes"] / (1024 * 1024)
        display_df["bytes_per_element"] = display_df["bytes_per_element"].round(2)
        st.dataframe(
            display_df[["representation", "elements", "payload_bytes", "payload_MiB", "bytes_per_element", "measurement"]],
            use_container_width=True,
            hide_index=True,
        )
        plot_df = display_df.copy()
        fig = px.bar(
            plot_df,
            x="representation",
            y="payload_MiB",
            color="representation",
            title="Measured storage footprint (MB)",
            labels={"representation": "Representation", "payload_MiB": "Memory (MiB)"},
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        smallest = memory_df.sort_values("bytes_per_element").iloc[0]
        st.info(f"Lowest measured bytes per element in this run: **{smallest['representation']}** ({smallest['bytes_per_element']:.2f} bytes/element).")
    else:
        st.info("Choose at least one dtype and run the memory benchmark to populate this section.")

with speed_tab:
    st.subheader("Execution time comparison")
    st.write(
        "Add, multiply and comparison operations reuse preallocated output arrays, so their "
        "timings exclude repeated output allocation. Reduction uses a float64 accumulator "
        "for a consistent accumulation policy. Each operation is warmed up, then timed "
        "across repeated batches; the median is reported."
    )
    if st.button("Run execution benchmark", type="primary", disabled=not chosen_speed, key="run_speed"):
        with st.spinner("Running timed operations. Larger datasets and more repeats take longer..."):
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
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        fastest = chart_df.sort_values("median_ms").iloc[0]
        st.info(f"Fastest dtype for this operation in this run: **{fastest['dtype']}** at {fastest['median_ms']:.4f} ms. This is machine- and workload-dependent.")
    else:
        st.info("Choose numeric dtypes and run the execution benchmark to populate this section.")

with precision_tab:
    st.subheader("Floating-point precision comparison")
    st.write(
        "Values are generated once in float64 and cast to each target format. Cast error "
        "is measured against the float64 reference. The dtype-specific sum is also compared "
        "with a float64 reference sum. The sample is capped at 10,000 elements to keep the "
        "float16 accumulation experiment within a safe numeric range."
    )
    if st.button("Run precision benchmark", type="primary", key="run_precision"):
        with st.spinner("Comparing floating-point precision..."):
            st.session_state.precision_results = precision_benchmark(dataset_size)
    if st.session_state.precision_results:
        precision_df = pd.DataFrame(st.session_state.precision_results)
        st.dataframe(precision_df.round(10), use_container_width=True, hide_index=True)
        fig = px.bar(
            precision_df,
            x="dtype",
            y="mean_abs_cast_error",
            title="Mean absolute representation error",
            labels={"mean_abs_cast_error": "Mean absolute error", "dtype": "Floating-point dtype"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the precision benchmark to compare float16, float32 and float64.")

with advisor_tab:
    st.subheader("Integer data type advisor")
    st.write("Find the smallest standard NumPy integer type that can represent an inclusive value range without overflow.")
    c1, c2 = st.columns(2)
    minimum = c1.number_input("Minimum value", value=-100, step=1, format="%d")
    maximum = c2.number_input("Maximum value", value=100, step=1, format="%d")
    if minimum <= maximum:
        recommendation = recommend_integer_dtype(int(minimum), int(maximum))
        if recommendation:
            st.success(f"Recommended integer dtype: **{recommendation}** ({np.dtype(recommendation).itemsize} byte(s) per element).")
        else:
            st.warning("No standard NumPy integer dtype can represent this range.")
    else:
        st.error("Minimum value must be less than or equal to maximum value.")
    st.caption("This advisor checks range safety only. Choose a wider type when future values may exceed the current range.")

with methodology_tab:
    st.subheader("How to interpret the measurements")
    st.markdown(
        """
        - **Memory:** `ndarray.nbytes` is the data-buffer size. Container/header overhead is reported separately. Python list estimates include the list and the shallow sizes of its element objects.
        - **Timing:** arrays are created before timing. One warm-up call is excluded. The median across repeats is the primary statistic; minimum and standard deviation are also reported.
        - **Arithmetic safety:** generated integer values are deliberately small so fixed-width add and multiply operations do not overflow during the speed test.
        - **Precision:** the float64-generated reference is treated as the comparison baseline. Floating-point results can differ due to rounding and accumulation order.
        - **Environment:** CPU, operating system, Python/NumPy versions, background processes, power mode, and memory bandwidth can affect results. Run multiple times under similar conditions when comparing changes.
        - **Scope:** this is a NumPy-focused microbenchmark, not a universal ranking of dtypes or hardware. A faster dtype in one operation may not be faster in another.
        """
    )
    st.json(info)

st.divider()
st.subheader("Export this experiment")
all_memory = st.session_state.memory_results
all_speed = st.session_state.speed_results
all_precision = st.session_state.precision_results
if any((all_memory, all_speed, all_precision)):
    st.download_button(
        "Download combined results (CSV)",
        data=results_bundle_csv(all_memory, all_speed, all_precision),
        file_name="datatype_lab_results.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download environment metadata (JSON)",
        data=metadata_json({
            "dataset_size": dataset_size,
            "memory_dtypes": chosen_memory,
            "speed_dtypes": chosen_speed,
            "timing_repeats": repeats,
            "iterations_per_repeat": iterations,
            "python_list_limit": 100000,
        }),
        file_name="benchmark_environment.json",
        mime="application/json",
    )
else:
    st.caption("Run at least one experiment to enable CSV and environment exports.")

st.caption("DataType Lab · Results are generated locally when you run each experiment.")
