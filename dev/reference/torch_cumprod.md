# Cumprod

Cumprod

## Usage

``` r
torch_cumprod(self, dim, dtype = NULL)
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

## cumprod(input, dim, out=NULL, dtype=NULL) -\> Tensor

Returns the cumulative product of elements of `input` in the dimension
`dim`.

For example, if `input` is a vector of size N, the result will also be a
vector of size N, with elements.

\$\$ y_i = x_1 \times x_2\times x_3\times \dots \times x_i \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(10))
a
torch_cumprod(a, dim=1)
}
#> torch_tensor
#> -1.1577
#>  2.0117
#>  2.7194
#>  0.2309
#> -0.0933
#> -0.0219
#> -0.0013
#>  0.0025
#>  0.0007
#>  0.0009
#> [ CPUFloatType{10} ]
```
