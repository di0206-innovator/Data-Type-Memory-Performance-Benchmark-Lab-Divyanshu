# Data Type Memory Performance Benchmark Lab

**Project type:** AI/ML laboratory project  
**Implementation:** Python, NumPy, Streamlit, Plotly  
**Student:** Divyanshu Sinha  
**Course / Section:** Essentials Of A  
**Instructor:** [Enter instructor name]  
**Submission date:** [Enter date]

---

## Abstract

This project develops an interactive benchmarking laboratory to study how data type selection affects memory footprint, execution time, and floating-point numerical precision. The application compares common NumPy Boolean, signed integer, unsigned integer, and floating-point formats, alongside selected Python list representations. It runs repeatable arithmetic and reduction microbenchmarks, measures floating-point representation and summation error, and provides an integer data type advisor. The application records the software and platform environment and exports experiment results for inspection. The project demonstrates why data type selection is an engineering trade-off among storage requirements, computational behaviour, and numerical accuracy rather than a single universal ranking.

## 1. Introduction

Data types define how values are represented and what range and precision can be stored. In data-intensive and AI/ML workflows, choosing an unnecessarily wide type can increase memory consumption and memory traffic. Choosing a type that is too narrow can cause overflow or unacceptable numerical error. The impact also depends on the operation, hardware, software implementation, and dataset.

This project provides a small experimental environment in which these trade-offs can be measured rather than assumed.

## 2. Problem statement

To design and implement a Python-based laboratory that measures and compares memory usage, execution time, and numerical precision for different data types under controlled, repeatable workloads.

## 3. Objectives

1. Measure the raw memory footprint of common NumPy data types for a fixed number of elements.
2. Contrast contiguous NumPy arrays with selected Python list representations.
3. Benchmark representative element-wise and reduction operations.
4. Quantify floating-point casting and summation error for float16, float32, and float64.
5. Recommend a safe integer dtype for a specified minimum and maximum value.
6. Export experiment results and environment metadata so findings can be reproduced and reported.

## 4. Tools and technologies

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Typed arrays and numerical operations |
| Pandas | Tabular result handling and CSV export |
| Streamlit | Interactive laboratory interface |
| Plotly | Interactive charts |
| `time.perf_counter_ns` | High-resolution elapsed-time measurement |
| `sys.getsizeof` | Shallow Python object/container size estimates |

## 5. Data types included

| Category | Data types |
|---|---|
| Boolean | bool |
| Signed integer | int8, int16, int32, int64 |
| Unsigned integer | uint8, uint16, uint32, uint64 |
| Floating point | float16, float32, float64 |
| Python reference representations | list[int], list[str] |

The Python list rows are separate representation comparisons. They should not be described as NumPy dtypes.

## 6. Experimental methodology

### 6.1 Memory experiment

For every selected NumPy dtype, a deterministic array of the configured size is created. Its data-buffer footprint is measured using `ndarray.nbytes`. The ndarray object's size is also displayed separately. For Python lists, the estimate is the list container's shallow size plus the shallow sizes of contained objects. The Python list sample is capped at 100,000 elements to keep the experiment responsive.

For a NumPy array:

\[
\text{Payload bytes} = N \times \text{dtype.itemsize}
\]

where \(N\) is the number of elements.

### 6.2 Execution-time experiment

The application benchmarks element-wise addition, element-wise multiplication, comparison, and summation. Arrays and output buffers are prepared before the timed region. Add, multiply, and comparison reuse preallocated buffers. Sum uses a float64 accumulator to keep the accumulation policy consistent across input types. Each function receives one warm-up call. The application then records multiple batches and reports the median, minimum, population standard deviation, and nanoseconds per element.

\[
\text{ns per element} = \frac{\text{median time in ns}}{N}
\]

Integer test values are intentionally small to avoid overflow in the add and multiply workloads.

### 6.3 Precision experiment

A deterministic reference array is generated in float64 and cast to float16, float32, and float64. The application reports mean and maximum absolute element-wise cast error. It also compares a dtype-specific sum with the float64 reference sum. The sample is capped at 10,000 values to keep the float16 sum within a safe range for this experiment.

\[
\text{Absolute error} = |x_{\text{typed}} - x_{\text{reference}}|
\]

### 6.4 Environment controls

The export records Python version, NumPy version, operating platform, and processor string when available. For fair comparisons, runs should use the same machine, dataset size, software environment, power mode, and background workload. Multiple runs are recommended.

## 7. Application modules

- **Memory profiler:** Runs the storage-footprint experiment and visualizes bytes / MiB.
- **Execution profiler:** Measures operation timings across selected numeric types.
- **Precision lab:** Reports float casting and summation errors.
- **Type advisor:** Selects the smallest standard NumPy integer type that can safely contain an inclusive range.
- **Export:** Produces a combined CSV and JSON environment metadata file.

## 8. Results and observations

> **Important: Replace this section with results from your own run. Do not submit invented measurements.**

### 8.1 Memory results

Attach the exported `datatype_lab_results.csv` or paste a screenshot/table from the Memory profiler.

| Representation | Elements | Measured payload bytes | Bytes per element |
|---|---:|---:|---:|
| NumPy int8 | [enter] | [enter] | [enter] |
| NumPy int32 | [enter] | [enter] | [enter] |
| NumPy int64 | [enter] | [enter] | [enter] |
| NumPy float32 | [enter] | [enter] | [enter] |
| NumPy float64 | [enter] | [enter] | [enter] |

**Observation:** [Explain how storage scales with item size and distinguish NumPy's contiguous payload from Python object/list estimates.]

### 8.2 Execution-time results

Insert the exported speed results or screenshots. Discuss each operation separately; do not assume one dtype is fastest for every workload.

**Observation:** [Identify the fastest and slowest measured dtypes for each operation on your machine. Include median times and mention run-to-run variation.]

### 8.3 Precision results

Insert the precision table from the application.

**Observation:** [Compare float16, float32 and float64 casting errors and sum errors. Explain the precision-versus-storage trade-off.]

### 8.4 Environment

| Item | Measured environment |
|---|---|
| Python | [from environment JSON] |
| NumPy | [from environment JSON] |
| Platform | [from environment JSON] |
| Processor | [from environment JSON] |
| Dataset size | [enter] |
| Timing repeats / iterations | [enter] |

## 9. Discussion

The theoretical storage requirement of a fixed-width NumPy array is determined by its element count and dtype item size. However, lower memory use does not automatically imply faster execution. Performance can be affected by CPU instruction support, vectorization, memory bandwidth, alignment, library build options, and operation type. Floating-point formats also differ in representable precision and range. Consequently, a type should be selected according to the valid data range, acceptable error, memory budget, and target operations.

## 10. Conclusion

The Data Type Memory Performance Benchmark Lab provides a repeatable way to investigate the relationship between data representation, memory usage, execution time, and precision. It combines direct storage measurements, operation-level timing, precision analysis, and range-based integer type selection in one application. The central conclusion should be based on the measurements collected on the student's machine: no data type is universally optimal; the appropriate choice depends on workload requirements and correctness constraints.

## 11. Limitations and future scope

The current timing tests are microbenchmarks and can be influenced by background processes and system state. Python object memory is estimated from shallow object sizes. The project currently focuses on NumPy arrays and selected Python lists rather than all dataframe, tensor, disk, or GPU representations. Future work can include isolated process peak-RAM measurement, matrix multiplication, sorting, serialization, pandas and PyTorch tensor comparisons, GPU memory profiling, and controlled CPU-affinity experiments.

## 12. References

1. NumPy documentation — Array objects and `ndarray.nbytes`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html
2. NumPy documentation — Data type objects (`dtype`): https://numpy.org/doc/stable/reference/arrays.dtypes.html
3. Python documentation — `sys.getsizeof`: https://docs.python.org/3/library/sys.html#sys.getsizeof
4. Python documentation — `time.perf_counter_ns`: https://docs.python.org/3/library/time.html#time.perf_counter_ns
5. Streamlit documentation: https://docs.streamlit.io/
