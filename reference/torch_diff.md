# Computes the n-th forward difference along the given dimension.

The first-order differences are given by
`out[i] = input[i + 1] - input[i]`. Higher-order differences are
calculated by using `torch_diff()` recursively.

## Usage

``` r
torch_diff(self, n = 1L, dim = -1L, prepend = list(), append = list())
```

## Arguments

- self:

  the tensor to compute the differences on

- n:

  the number of times to recursively compute the difference

- dim:

  the dimension to compute the difference along. Default is the last
  dimension.

- prepend:

  values to prepend to input along dim before computing the difference.
  Their dimensions must be equivalent to that of input, and their shapes
  must match input’s shape except on dim.

- append:

  values to append to input along dim before computing the difference.
  Their dimensions must be equivalent to that of input, and their shapes
  must match input’s shape except on dim.

## Note

Only n = 1 is currently supported

## Examples

``` r
if (torch_is_installed()) {
a <- torch_tensor(c(1,2,3))
torch_diff(a)

b <- torch_tensor(c(4, 5))
torch_diff(a, append = b)

c <- torch_tensor(rbind(c(1,2,3), c(3,4,5)))
torch_diff(c, dim = 1)
torch_diff(c, dim = 2) 

}
#> torch_tensor
#>  1  1
#>  1  1
#> [ CPUFloatType{2,2} ]
```
