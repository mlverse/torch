# Narrow

Narrow

## Usage

``` r
torch_narrow(self, dim, start, length)
```

## Arguments

- self:

  (Tensor) the tensor to narrow

- dim:

  (int) the dimension along which to narrow

- start:

  (int) the starting dimension

- length:

  (int) the distance to the ending dimension

## narrow(input, dim, start, length) -\> Tensor

Returns a new tensor that is a narrowed version of `input` tensor. The
dimension `dim` is input from `start` to `start + length`. The returned
tensor and `input` tensor share the same underlying storage.

## Examples

``` r
if (torch_is_installed()) {

x = torch_tensor(matrix(c(1:9), ncol = 3, byrow= TRUE))
torch_narrow(x, 1, 1, 2)
torch_narrow(x, 2, 2, 2)
}
#> torch_tensor
#>  2  3
#>  5  6
#>  8  9
#> [ CPULongType{3,2} ]
```
