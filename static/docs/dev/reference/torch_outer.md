# Outer

Outer

## Usage

``` r
torch_outer(self, vec2)
```

## Arguments

- self:

  (Tensor) 1-D input vector

- vec2:

  (Tensor) 1-D input vector

## Note

This function does not broadcast.

## outer(input, vec2, \*, out=None) -\> Tensor

Outer product of `input` and `vec2`. If `input` is a vector of size
\\n\\ and `vec2` is a vector of size \\m\\, then `out` must be a matrix
of size \\(n \times m)\\.

## Examples

``` r
if (torch_is_installed()) {

v1 <- torch_arange(1., 5.)
v2 <- torch_arange(1., 4.)
torch_outer(v1, v2)
}
#> torch_tensor
#>   1   2   3   4
#>   2   4   6   8
#>   3   6   9  12
#>   4   8  12  16
#>   5  10  15  20
#> [ CPUFloatType{5,4} ]
```
