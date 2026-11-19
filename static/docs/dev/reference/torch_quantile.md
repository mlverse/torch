# Quantile

Quantile

## Usage

``` r
torch_quantile(self, q, dim = NULL, keepdim = FALSE, interpolation = "linear")
```

## Arguments

- self:

  (Tensor) the input tensor.

- q:

  (float or Tensor) a scalar or 1D tensor of quantile values in the
  range `[0, 1]`

- dim:

  (int) the dimension to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

- interpolation:

  The interpolation method.

## quantile(input, q) -\> Tensor

Returns the q-th quantiles of all elements in the `input` tensor, doing
a linear interpolation when the q-th quantile lies between two data
points.

## quantile(input, q, dim=None, keepdim=FALSE, \*, out=None) -\> Tensor

Returns the q-th quantiles of each row of the `input` tensor along the
dimension `dim`, doing a linear interpolation when the q-th quantile
lies between two data points. By default, `dim` is `None` resulting in
the `input` tensor being flattened before computation.

If `keepdim` is `TRUE`, the output dimensions are of the same size as
`input` except in the dimensions being reduced (`dim` or all if `dim` is
`NULL`) where they have size 1. Otherwise, the dimensions being reduced
are squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)).
If `q` is a 1D tensor, an extra dimension is prepended to the output
tensor with the same size as `q` which represents the quantiles.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_randn(c(1, 3))
a
q <- torch_tensor(c(0, 0.5, 1))
torch_quantile(a, q)


a <- torch_randn(c(2, 3))
a
q <- torch_tensor(c(0.25, 0.5, 0.75))
torch_quantile(a, q, dim=1, keepdim=TRUE)
torch_quantile(a, q, dim=1, keepdim=TRUE)$shape
}
#> [1] 3 1 3
```
