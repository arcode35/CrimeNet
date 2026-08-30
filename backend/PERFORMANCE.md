# CrimeNet serving performance

## Interactive mark inference

`MarkRuntime` benchmarks 300 uncached, single-row predictions against the actual
loaded production model at process startup. It logs p50/p95/p99 latency,
throughput, CPU/GPU numerical agreement, and the selected path under
`mark_inference_benchmark`. CPU is selected when its p95 is within 10% of GPU;
otherwise the GPU runs through a bounded 3 ms microbatch queue.

Run the same benchmark explicitly on a serving host with:

```shell
python backend/benchmark_mark_inference.py
```

`CRIMENET_MARK_DEVICE=cpu|gpu|auto` controls selection. A forced unavailable GPU
fails startup rather than silently changing the requested configuration.

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
