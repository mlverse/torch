# Nanquantile

Nanquantile

## Usage

``` r
torch_nanquantile(
  self,
  q,
  dim = NULL,
  keepdim = FALSE,
  interpolation = "linear"
)
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

## nanquantile(input, q, dim=None, keepdim=FALSE, \*, out=None) -\> Tensor

This is a variant of
[`torch_quantile()`](https://torch.mlverse.org/docs/dev/reference/torch_quantile.md)
that "ignores" `NaN` values, computing the quantiles `q` as if `NaN`
values in `input` did not exist. If all values in a reduced row are
`NaN` then the quantiles for that reduction will be `NaN`. See the
documentation for
[`torch_quantile()`](https://torch.mlverse.org/docs/dev/reference/torch_quantile.md).

## Examples

``` r
if (torch_is_installed()) {

t <- torch_tensor(c(NaN, 1, 2))
t$quantile(0.5)
t$nanquantile(0.5)
t <- torch_tensor(rbind(c(NaN, NaN), c(1, 2)))
t
t$nanquantile(0.5, dim=1)
t$nanquantile(0.5, dim=2)
torch_nanquantile(t, 0.5, dim = 1)
torch_nanquantile(t, 0.5, dim = 2)
}
#> torch_tensor
#>     nan  1.5000
#> [ CPUFloatType{1,2} ]
```
