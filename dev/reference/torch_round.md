# Round

Round

## Usage

``` r
torch_round(self, decimals)
```

## Arguments

- self:

  (Tensor) the input tensor.

- decimals:

  Number of decimal places to round to (default: 0). If decimals is
  negative, it specifies the number of positions to the left of the
  decimal point.

## round(input, out=NULL) -\> Tensor

Returns a new tensor with each of the elements of `input` rounded to the
closest integer.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_round(a)
}
#> torch_tensor
#>  2
#>  1
#> -3
#>  1
#> [ CPUFloatType{4} ]
```
