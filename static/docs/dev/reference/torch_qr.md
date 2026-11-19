# Qr

Qr

## Usage

``` r
torch_qr(self, some = TRUE)
```

## Arguments

- self:

  (Tensor) the input tensor of size \\(\*, m, n)\\ where `*` is zero or
  more batch dimensions consisting of matrices of dimension \\m \times
  n\\.

- some:

  (bool, optional) Set to `TRUE` for reduced QR decomposition and
  `FALSE` for complete QR decomposition.

## Note

precision may be lost if the magnitudes of the elements of `input` are
large

While it should always give you a valid decomposition, it may not give
you the same one across platforms - it will depend on your LAPACK
implementation.

## qr(input, some=TRUE, out=NULL) -\> (Tensor, Tensor)

Computes the QR decomposition of a matrix or a batch of matrices
`input`, and returns a namedtuple (Q, R) of tensors such that
\\\mbox{input} = Q R\\ with \\Q\\ being an orthogonal matrix or batch of
orthogonal matrices and \\R\\ being an upper triangular matrix or batch
of upper triangular matrices.

If `some` is `TRUE`, then this function returns the thin (reduced) QR
factorization. Otherwise, if `some` is `FALSE`, this function returns
the complete QR factorization.

## Examples

``` r
if (torch_is_installed()) {

a = torch_tensor(matrix(c(12., -51, 4, 6, 167, -68, -4, 24, -41), ncol = 3, byrow = TRUE))
out = torch_qr(a)
q = out[[1]]
r = out[[2]]
torch_mm(q, r)$round()
torch_mm(q$t(), q)$round()
}
#> torch_tensor
#>  1 -0  0
#> -0  1  0
#>  0  0  1
#> [ CPUFloatType{3,3} ]
```
