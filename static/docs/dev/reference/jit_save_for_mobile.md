# Saves a `script_function` or `script_module` in bytecode form, to be loaded on a mobile device

Saves a `script_function` or `script_module` in bytecode form, to be
loaded on a mobile device

## Usage

``` r
jit_save_for_mobile(obj, path, ...)
```

## Arguments

- obj:

  An `script_function` or `script_module` to save

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
jit_save_for_mobile(tr_fn, tmp)
}
```
