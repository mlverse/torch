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

## Fix 2: Replace std::set with linear scan in type resolution

The `expected_types` vectors are tiny (1-3 elements). Building a `std::set`
from them on every call is slower than a simple `std::find` linear scan.
Also removed a dead debug loop in `create_fn_name`.

| expression     | median (before) | median (after) | change |
|----------------|-----------------|----------------|--------|
| dispatch       | 6.11µs          | 5.90µs         | -3.4%  |
| create_fn_name | 1.39µs          | 1.19µs         | -14.4% |
| direct         | 1.52µs          | 1.56µs         | (noise)|

## Fix 3: Single-pass cpp_clean_names with lookup table

Replace 11 sequential `erase(std::remove(...))` passes with a single pass
using a static `bool[256]` lookup table. Also removed the dead
`remove_characters` vector.

| expression     | median (before) | median (after) | change |
|----------------|-----------------|----------------|--------|
| dispatch       | 5.90µs          | 5.86µs         | (noise)|
| create_fn_name | 1.19µs          | 1.27µs         | (noise)|

Negligible impact — the generated function names are short enough that the
multi-pass approach was already fast. Still a cleaner implementation.
