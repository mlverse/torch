# Diag_embed

Diag_embed

## Usage

``` r
torch_diag_embed(self, offset = 0L, dim1 = -2L, dim2 = -1L)
```

## Arguments

- self:

  (Tensor) the input tensor. Must be at least 1-dimensional.

- offset:

  (int, optional) which diagonal to consider. Default: 0 (main
  diagonal).

- dim1:

  (int, optional) first dimension with respect to which to take
  diagonal. Default: -2.

- dim2:

  (int, optional) second dimension with respect to which to take
  diagonal. Default: -1.

## diag_embed(input, offset=0, dim1=-2, dim2=-1) -\> Tensor

Creates a tensor whose diagonals of certain 2D planes (specified by
`dim1` and `dim2`) are filled by `input`. To facilitate creating batched
diagonal matrices, the 2D planes formed by the last two dimensions of
the returned tensor are chosen by default.

The argument `offset` controls which diagonal to consider:

- If `offset` = 0, it is the main diagonal.

- If `offset` \> 0, it is above the main diagonal.

- If `offset` \< 0, it is below the main diagonal.

The size of the new matrix will be calculated to make the specified
diagonal of the size of the last input dimension. Note that for `offset`
other than \\0\\, the order of `dim1` and `dim2` matters. Exchanging
them is equivalent to changing the sign of `offset`.

Applying `torch_diagonal` to the output of this function with the same
arguments yields a matrix identical to input. However, `torch_diagonal`
has different default dimensions, so those need to be explicitly
specified.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(2, 3))
torch_diag_embed(a)
torch_diag_embed(a, offset=1, dim1=1, dim2=3)
}
#> torch_tensor
#> (1,.,.) = 
#>   0.0000  0.4175  0.0000  0.0000
#>   0.0000  1.3978  0.0000  0.0000
#> 
#> (2,.,.) = 
#>   0.0000  0.0000  0.6868  0.0000
#>   0.0000  0.0000 -0.6773  0.0000
#> 
#> (3,.,.) = 
#>   0.0000  0.0000  0.0000  2.3290
#>   0.0000  0.0000  0.0000 -0.0755
#> 
#> (4,.,.) = 
#>   0  0  0  0
#>   0  0  0  0
#> [ CPUFloatType{4,2,4} ]
```
