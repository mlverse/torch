# Isfinite

Isfinite

## Usage

``` r
torch_isfinite(self)
```

## Arguments

- self:

  (Tensor) A tensor to check

## TEST

Returns a new tensor with boolean elements representing if each element
is `Finite` or not.

## Examples

``` r
if (torch_is_installed()) {

torch_isfinite(torch_tensor(c(1, Inf, 2, -Inf, NaN)))
}
#> torch_tensor
#>  1
#>  0
#>  1
#>  0
#>  0
#> [ CPUBoolType{5} ]
```
