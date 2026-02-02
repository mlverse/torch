# Cholesky

Cholesky

## Usage

``` r
torch_cholesky(self, upper = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor \\A\\ of size \\(\*, n, n)\\ where `*` is
  zero or more batch dimensions consisting of symmetric
  positive-definite matrices.

- upper:

  (bool, optional) flag that indicates whether to return a upper or
  lower triangular matrix. Default: `FALSE`

## cholesky(input, upper=False, out=NULL) -\> Tensor

Computes the Cholesky decomposition of a symmetric positive-definite
matrix \\A\\ or for batches of symmetric positive-definite matrices.

If `upper` is `TRUE`, the returned matrix `U` is upper-triangular, and
the decomposition has the form:

\$\$ A = U^TU \$\$ If `upper` is `FALSE`, the returned matrix `L` is
lower-triangular, and the decomposition has the form:

\$\$ A = LL^T \$\$ If `upper` is `TRUE`, and \\A\\ is a batch of
symmetric positive-definite matrices, then the returned tensor will be
composed of upper-triangular Cholesky factors of each of the individual
matrices. Similarly, when `upper` is `FALSE`, the returned tensor will
be composed of lower-triangular Cholesky factors of each of the
individual matrices.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a = torch_mm(a, a$t()) # make symmetric positive-definite
l = torch_cholesky(a)
a
l
torch_mm(l, l$t())
a = torch_randn(c(3, 2, 2))
if (FALSE) { # \dontrun{
a = torch_matmul(a, a$transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
l = torch_cholesky(a)
z = torch_matmul(l, l$transpose(-1, -2))
torch_max(torch_abs(z - a)) # Max non-zero
} # }
}
```
