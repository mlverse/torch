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
#> -1.1738 -1.5019  0.6754  0.3351 -0.9309  1.1049
#>  0.1467  0.0893  1.0701  0.0803 -0.9897  1.5285
#>  0.0000 -0.2785 -0.9613 -1.8354  1.0776 -0.2361
#>  0.0000  0.0000  1.5773 -2.0471 -0.8265  0.5226
#> [ CPUFloatType{4,6} ]
```
