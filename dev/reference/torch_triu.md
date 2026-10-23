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
#> -0.8893  0.7502  0.3129 -0.4807  1.2804  0.8710
#>  0.4414 -1.3726  0.5171  1.6071  0.6334  0.0097
#>  0.0000  0.7820 -1.5171 -0.1044 -0.9254 -0.1198
#>  0.0000  0.0000 -0.2311 -0.2007 -0.2410  0.4488
#> [ CPUFloatType{4,6} ]
```
