# Flipud

Flipud

## Usage

``` r
torch_flipud(self)
```

## Arguments

- self:

  (Tensor) Must be at least 1-dimensional.

## Note

Equivalent to `input[-1,]`. Requires the array to be at least 1-D.

## flipud(input) -\> Tensor

Flip array in the up/down direction, returning a new tensor.

Flip the entries in each column in the up/down direction. Rows are
preserved, but appear in a different order than before.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_arange(start = 1, end = 4)$view(c(2, 2))
x
torch_flipud(x)
}
#> torch_tensor
#>  3  4
#>  1  2
#> [ CPUFloatType{2,2} ]
```
