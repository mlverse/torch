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
#>  0.3237
#> -0.5801
#>  0.1684
#>  0.8178
#>  0.9268
#>  2.6237
#>  4.7017
#>  5.5915
#>  5.6092
#>  6.9167
#> [ CPUFloatType{10} ]
```
