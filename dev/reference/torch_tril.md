# Tril

Tril

## Usage

``` r
torch_tril(self, diagonal = 0L)
```

## Arguments

- self:

  (Tensor) the input tensor.

- diagonal:

  (int, optional) the diagonal to consider

## tril(input, diagonal=0, out=NULL) -\> Tensor

Returns the lower triangular part of the matrix (2-D tensor) or batch of
matrices `input`, the other elements of the result tensor `out` are set
to 0.

The lower triangular part of the matrix is defined as the elements on
and below the diagonal.

The argument `diagonal` controls which diagonal to consider. If
`diagonal` = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the
main diagonal, and similarly a negative value excludes just as many
diagonals below the main diagonal. The main diagonal are the set of
indices \\\lbrace (i, i) \rbrace\\ for \\i \in \[0, \min\\d\_{1},
d\_{2}\\ - 1\]\\ where \\d\_{1}, d\_{2}\\ are the dimensions of the
matrix.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a
torch_tril(a)
b = torch_randn(c(4, 6))
b
torch_tril(b, diagonal=1)
torch_tril(b, diagonal=-1)
}
#> torch_tensor
#>  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
#>  1.5434  0.0000  0.0000  0.0000  0.0000  0.0000
#>  0.0262 -0.7762  0.0000  0.0000  0.0000  0.0000
#> -0.7777 -1.0096 -1.2715  0.0000  0.0000  0.0000
#> [ CPUFloatType{4,6} ]
```
