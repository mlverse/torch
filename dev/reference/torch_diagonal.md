# Diagonal

Diagonal

## Usage

``` r
torch_diagonal(self, outdim, dim1 = 1L, dim2 = 2L, offset = 0L)
```

## Arguments

- self:

  (Tensor) the input tensor. Must be at least 2-dimensional.

- outdim:

  dimension name if `self` is a named tensor.

- dim1:

  (int, optional) first dimension with respect to which to take
  diagonal. Default: 0.

- dim2:

  (int, optional) second dimension with respect to which to take
  diagonal. Default: 1.

- offset:

  (int, optional) which diagonal to consider. Default: 0 (main
  diagonal).

## diagonal(input, offset=0, dim1=0, dim2=1) -\> Tensor

Returns a partial view of `input` with the its diagonal elements with
respect to `dim1` and `dim2` appended as a dimension at the end of the
shape.

The argument `offset` controls which diagonal to consider:

- If `offset` = 0, it is the main diagonal.

- If `offset` \> 0, it is above the main diagonal.

- If `offset` \< 0, it is below the main diagonal.

Applying `torch_diag_embed` to the output of this function with the same
arguments yields a diagonal matrix with the diagonal entries of the
input. However, `torch_diag_embed` has different default dimensions, so
those need to be explicitly specified.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a
torch_diagonal(a, offset = 0)
torch_diagonal(a, offset = 1)
x = torch_randn(c(2, 5, 4, 2))
torch_diagonal(x, offset=-1, dim1=1, dim2=2)
}
#> torch_tensor
#> (1,.,.) = 
#>  -0.1885
#>  -1.0867
#> 
#> (2,.,.) = 
#>  0.01 *
#>  -0.4276
#>    7.8076
#> 
#> (3,.,.) = 
#>  -0.4325
#>   0.5643
#> 
#> (4,.,.) = 
#>   0.9335
#>   0.4813
#> [ CPUFloatType{4,2,1} ]
```
