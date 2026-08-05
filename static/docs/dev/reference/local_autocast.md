# Autocast context manager

Allow regions of your code to run in mixed precision. In these regions,
ops run in an op-specific dtype chosen by autocast to improve
performance while maintaining accuracy.

## Usage

``` r
local_autocast(
  device_type,
  dtype = NULL,
  enabled = TRUE,
  cache_enabled = NULL,
  ...,
  .env = parent.frame()
)

with_autocast(
  code,
  ...,
  device_type,
  dtype = NULL,
  enabled = TRUE,
  cache_enabled = NULL
)

set_autocast(device_type, dtype = NULL, enabled = TRUE, cache_enabled = NULL)

unset_autocast(context)
```

## Arguments

- device_type:

  a character string indicating whether to use 'cuda' or 'cpu' device

- dtype:

  a torch data type indicating whether to use
  [`torch_float16()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md)
  or
  [`torch_bfloat16()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md).

- enabled:

  a logical value indicating whether autocasting should be enabled in
  the region. Default: TRUE

- cache_enabled:

  a logical value indicating whether the weight cache inside autocast
  should be enabled.

- ...:

  currently unused.

- .env:

  The environment to use for scoping.

- code:

  code to be executed with no gradient recording.

- context:

  Returned by `set_autocast` and should be passed when unsetting it.

## Details

When entering an autocast-enabled region, Tensors may be any type. You
should not call `half()` or `bfloat16()` on your model(s) or inputs when
using autocasting.

`autocast` should only be enabled during the forward pass(es) of your
network, including the loss computation(s). Backward passes under
autocast are not recommended. Backward ops run in the same type that
autocast used for corresponding forward ops.

## Functions

- `with_autocast()`: A with context for automatic mixed precision.

- `set_autocast()`: Set the autocast context. For advanced users only.

- `unset_autocast()`: Unset the autocast context.

## See also

[`cuda_amp_grad_scaler()`](https://torch.mlverse.org/docs/dev/reference/cuda_amp_grad_scaler.md)
to perform dynamic gradient scaling.

## Examples

``` r
if (torch_is_installed()) {
x <- torch_randn(5, 5, dtype = torch_float32())
y <- torch_randn(5, 5, dtype = torch_float32())

foo <- function(x, y) {
  local_autocast(device = "cpu")
  z <- torch_mm(x, y)
  w <- torch_mm(z, x)
  w
}

out <- foo(x, y)
}
```
