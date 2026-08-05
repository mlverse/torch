# Isinf

Isinf

## Usage

``` r
torch_isinf(self)
```

## Arguments

- self:

  (Tensor) A tensor to check

## TEST

Returns a new tensor with boolean elements representing if each element
is `+/-INF` or not.

## Examples

``` r
if (torch_is_installed()) {

torch_isinf(torch_tensor(c(1, Inf, 2, -Inf, NaN)))
}
#> torch_tensor
#>  0
#>  1
#>  0
#>  1
#>  0
#> [ CPUBoolType{5} ]
```
