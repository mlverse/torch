# Log1p

Log1p

## Usage

``` r
torch_log1p(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## Note

This function is more accurate than
[`torch_log`](https://torch.mlverse.org/docs/dev/reference/torch_log.md)
for small values of `input`

## log1p(input, out=NULL) -\> Tensor

Returns a new tensor with the natural logarithm of (1 + `input`).

\$\$ y_i = \log\_{e} (x_i + 1) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(5))
a
torch_log1p(a)
}
#> torch_tensor
#>  0.0863
#>  0.5328
#> -2.5439
#> -0.1407
#> -0.0151
#> [ CPUFloatType{5} ]
```
