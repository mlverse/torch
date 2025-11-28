# Cumsum

Cumsum

## Usage

``` r
torch_cumsum(self, dim, dtype = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to do the operation over

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor. If
  specified, the input tensor is casted to `dtype` before the operation
  is performed. This is useful for preventing data type overflows.
  Default: NULL.

## cumsum(input, dim, out=NULL, dtype=NULL) -\> Tensor

Returns the cumulative sum of elements of `input` in the dimension
`dim`.

For example, if `input` is a vector of size N, the result will also be a
vector of size N, with elements.

\$\$ y_i = x_1 + x_2 + x_3 + \dots + x_i \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(10))
a
torch_cumsum(a, dim=1)
}
#> torch_tensor
#> -0.2825
#>  0.3570
#>  0.4234
#>  0.1212
#> -0.2598
#>  0.9732
#>  0.9539
#> -0.2055
#>  2.0442
#>  2.9938
#> [ CPUFloatType{10} ]
```
