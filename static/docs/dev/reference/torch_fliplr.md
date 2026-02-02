# Fliplr

Fliplr

## Usage

``` r
torch_fliplr(self)
```

## Arguments

- self:

  (Tensor) Must be at least 2-dimensional.

## Note

Equivalent to `input[,-1]`. Requires the array to be at least 2-D.

## fliplr(input) -\> Tensor

Flip array in the left/right direction, returning a new tensor.

Flip the entries in each row in the left/right direction. Columns are
preserved, but appear in a different order than before.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_arange(start = 1, end = 4)$view(c(2, 2))
x
torch_fliplr(x)
}
#> torch_tensor
#>  2  1
#>  4  3
#> [ CPUFloatType{2,2} ]
```
