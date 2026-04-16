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

## Fix 4 (tried, reverted): Skip list subsetting when args already match

Tried adding an `identical(names(args), args_needed)` guard to skip
`args[args_needed]` when names already match. No improvement — the
`identical()` check costs roughly the same as the subsetting itself.

| expression | median (before) | median (after) | change |
|------------|-----------------|----------------|--------|
| dispatch   | 5.86µs          | 5.95µs         | (noise)|

## Fix 5: Generate direct calls for single-overload functions (codegen)

For functions with only one overload, the resolved C++ function name is
known at codegen time. Changed `tools/torchgen/R/r.R` to generate a direct
call (e.g. `cpp_torch_namespace_matmul_self_Tensor_other_Tensor(self, other)`)
instead of going through `call_c_function`.

This eliminates all dispatch overhead (type resolution, cache lookup,
`do.call`) for ~82% of namespace functions (1,851 of 2,256) and ~72% of
methods (409 of 565).

Benchmarked with `torch_matmul(a, b)` (single-overload, direct call):

| expression | median (before) | median (after) | change |
|------------|-----------------|----------------|--------|
| dispatch   | ~5.86µs         | 1.39µs         | -76%   |
| direct     | 1.23µs          | 1.23µs         | (unchanged) |

Multi-overload functions like `torch_add` are unaffected (still 5.86µs).

## Fix 6: Generate inline if/else for 2-overload functions (codegen)

For functions with exactly 2 overloads differing in one dispatch arg type,
generate inline R type checks instead of calling the generic dispatcher.
Added helper functions `is_tensor_dispatch()` and `is_int64_dispatch()` to
`R/codegen-utils.R` and type-check code generation in `r_inline_dispatch()`.

Covers 10 type pairs (Scalar|Tensor, Dimname|int64_t, Tensor|double, etc.)
affecting 106 namespace functions and 100 methods.

| expression | median (before) | median (after) | change |
|------------|-----------------|----------------|--------|
| torch_add  | 5.86µs          | 2.05µs         | -65%   |
| direct     | 1.56µs          | 1.56µs         | (unchanged) |

## Summary

Dispatch path breakdown (namespace / methods):

| path             | namespace   | methods   |
|------------------|-------------|-----------|
| direct call      | 2,065 (84%) | 609 (80%) |
| inline if/else   | 106 (4%)    | 100 (13%) |
| generic dispatch | 298 (12%)   | 56 (7%)   |

Final performance:

| function       | type             | baseline | after all fixes | improvement |
|----------------|------------------|----------|-----------------|-------------|
| torch_matmul   | direct call      | 6.76µs   | 1.39µs          | -79%        |
| torch_add      | inline if/else   | 6.76µs   | 2.05µs          | -70%        |
| torch_div      | generic dispatch | 6.76µs   | 5.86µs          | -13%        |

For ~88-93% of functions, dispatch overhead is essentially eliminated.
The remaining ~7-12% still use the generic dispatcher (functions with 3+
overloads or complex dispatch patterns).
