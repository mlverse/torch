# Dispatch Performance Optimization Results

Benchmarking `torch_add(a, b)` with 2x2 tensors to isolate R-level dispatch overhead.

## Baseline

| expression     | median   | itr/sec  |
|----------------|----------|----------|
| dispatch       | 6.76µs   | 139,648  |
| create_fn_name | 1.39µs   | 687,513  |
| ns_lookup      | 0.29µs   | 2,949,012 |
| do_call        | 2.99µs   | 285,668  |
| direct         | 1.56µs   | 601,572  |

Dispatch overhead: ~5.2µs per call (dispatch - direct).
Biggest bottleneck: `do_call` (formals + do.call) at ~3µs.

## Fix 1: Cache function + formals lookup

Cache the resolved function and its `formals()` names in an environment
(`.dispatch_cache`) keyed by the resolved function name. Eliminates repeated
`getNamespace()` lookup and `formals()` parsing after the first call.

| expression | median (before) | median (after) | change |
|------------|-----------------|----------------|--------|
| dispatch   | 6.76µs          | 6.11µs         | -9.6%  |
| direct     | 1.56µs          | 1.52µs         | (noise)|
