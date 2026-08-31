# CrimeNet serving performance

## Interactive mark inference

Normal service startup is deterministic and performs zero benchmark predictions.
If `CRIMENET_MARK_INFERENCE` is unset, CrimeSense uses CPU serving. CPU mode does
not import CuPy, initialize CUDA, or create the GPU queue/worker.

```bash
# Recommended deterministic production configuration
export CRIMENET_MARK_INFERENCE=cpu

# Explicit GPU microbatch mode
export CRIMENET_MARK_INFERENCE=gpu_batch

# Diagnostic only — benchmarks CPU vs GPU during startup
export CRIMENET_MARK_INFERENCE=auto
```

`auto` should generally not be used in the systemd production service. It runs
300 uncached single-row predictions on both available runtimes, logs p50/p95/p99
latency, throughput, numerical agreement, and the recommended serving mode under
`mark_inference_benchmark`. CPU is recommended when its p95 is within 10% of GPU;
otherwise the bounded 3 ms GPU microbatch runtime is recommended.

Run the same benchmark explicitly on a serving host with:

```shell
python backend/benchmark_mark_inference.py
```

The standalone script explicitly selects diagnostic `auto` mode, regardless of
the production default. Explicit `gpu_batch` mode fails startup if CuPy/CUDA is
unavailable rather than silently changing serving behavior.

## Viewport GZip

Measured 2026-08-30 with a representative 25,000-cell JSON response (2,750,101
uncompressed bytes), 20 runs per level:

| level | median | p95 | bytes |
| --- | ---: | ---: | ---: |
| 1 | 3.523 ms | 3.733 ms | 76,197 |
| 6 | 8.717 ms | 9.063 ms | 75,243 |
| 9 | 10.408 ms | 10.831 ms | 74,561 |

Level 1 removes about 6.9 ms of compression CPU time versus the previous level 9
default while increasing this compressed response by about 2.2%, so the API keeps
GZip enabled and uses level 1.

## Viewport result construction

The old scalar loop and the vectorized implementation were each measured for 20
runs over 25,000 matched cells, with exact output equality asserted:

| implementation | median | p95 |
| --- | ---: | ---: |
| scalar loop | 8.498 ms | 9.839 ms |
| vectorized arithmetic | 7.153 ms | 7.920 ms |

That is a 1.19x median construction speedup. The larger win under burst traffic
comes from bounded admission, because obsolete requests can no longer occupy an
unbounded amount of concurrent H3 and response-building work.
