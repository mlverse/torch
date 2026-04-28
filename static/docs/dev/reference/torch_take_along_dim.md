# Selects values from input at the 1-dimensional indices from indices along the given dim.

Selects values from input at the 1-dimensional indices from indices
along the given dim.

## Usage

``` r
torch_take_along_dim(self, indices, dim = NULL)
```

## Arguments

- self:

  the input tensor.

- indices:

  the indices into input. Must have long dtype.

- dim:

  the dimension to select along. Default is `NULL`.

## Note

If dim is `NULL`, the input array is treated as if it has been flattened
to 1d.

Functions that return indices along a dimension, like
[`torch_argmax()`](https://torch.mlverse.org/docs/dev/reference/torch_argmax.md)
and
[`torch_argsort()`](https://torch.mlverse.org/docs/dev/reference/torch_argsort.md),
are designed to work with this function. See the examples below.

## Examples

``` r
if (torch_is_installed()) {
t <- torch_tensor(matrix(c(10, 30, 20, 60, 40, 50), nrow = 2))
max_idx <- torch_argmax(t)
torch_take_along_dim(t, max_idx)

sorted_idx <- torch_argsort(t, dim=2)
torch_take_along_dim(t, sorted_idx, dim=2)

}
#> torch_tensor
#>  10  20  40
#>  30  50  60
#> [ CPUFloatType{2,3} ]
```
