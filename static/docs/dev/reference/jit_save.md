# Saves a `script_function` to a path

Saves a `script_function` to a path

## Usage

``` r
jit_save(obj, path, ...)
```

## Arguments

- obj:

  An `script_function` to save

- path:

  The path to save the serialized function.

- ...:

  currently unused

## Examples

``` r
if (torch_is_installed()) {
fn <- function(x) {
  torch_relu(x)
}

input <- torch_tensor(c(-1, 0, 1))
tr_fn <- jit_trace(fn, input)

tmp <- tempfile("tst", fileext = "pt")
jit_save(tr_fn, tmp)
}
```
