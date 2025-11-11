# Isnan

Isnan

## Usage

``` r
torch_isnan(self)
```

## Arguments

- self:

  (Tensor) A tensor to check

## TEST

Returns a new tensor with boolean elements representing if each element
is `NaN` or not.

## Examples

``` r
if (torch_is_installed()) {

torch_isnan(torch_tensor(c(1, NaN, 2)))
}
#> torch_tensor
#>  0
#>  1
#>  0
#> [ CPUBoolType{3} ]
```
