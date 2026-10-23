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
#>  4.4996e-01
#>  4.9321e-01
#> -2.0562e-01
#> -2.4851e-03
#>  1.6249e-03
#> -1.5121e-03
#>  1.0092e-04
#> -7.8860e-05
#>  9.8478e-06
#>  4.3265e-06
#> [ CPUFloatType{10} ]
```
