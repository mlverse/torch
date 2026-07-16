# Triu

Triu

## Usage

``` r
torch_triu(self, diagonal = 0L)
```

## Arguments

- self:

  (Tensor) the input tensor.

- diagonal:

  (int, optional) the diagonal to consider

## triu(input, diagonal=0, out=NULL) -\> Tensor

Returns the upper triangular part of a matrix (2-D tensor) or batch of
matrices `input`, the other elements of the result tensor `out` are set
to 0.

The upper triangular part of the matrix is defined as the elements on
and above the diagonal.

The argument `diagonal` controls which diagonal to consider. If
`diagonal` = 0, all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the
main diagonal, and similarly a negative value includes just as many
diagonals below the main diagonal. The main diagonal are the set of
indices \\\lbrace (i, i) \rbrace\\ for \\i \in \[0, \min\\d\_{1},
d\_{2}\\ - 1\]\\ where \\d\_{1}, d\_{2}\\ are the dimensions of the
matrix.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a
torch_triu(a)
torch_triu(a, diagonal=1)
torch_triu(a, diagonal=-1)
b = torch_randn(c(4, 6))
b
torch_triu(b, diagonal=1)
torch_triu(b, diagonal=-1)
}
#> torch_tensor
#>  0.3810  0.3347 -0.4727 -1.7572  1.2074 -2.4219
#>  0.6497 -0.3450 -1.1122  0.2839 -0.0072 -0.3261
#>  0.0000  0.3539 -0.1050 -0.0032 -0.6996  0.4223
#>  0.0000  0.0000  0.2958 -1.3924  0.5336 -0.6111
#> [ CPUFloatType{4,6} ]
```
