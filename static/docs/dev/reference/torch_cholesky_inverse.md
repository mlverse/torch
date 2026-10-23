# Cholesky_inverse

Cholesky_inverse

## Usage

``` r
torch_cholesky_inverse(self, upper = FALSE)
```

## Arguments

- self:

  (Tensor) the input 2-D tensor \\u\\, a upper or lower triangular
  Cholesky factor

- upper:

  (bool, optional) whether to return a lower (default) or upper
  triangular matrix

## cholesky_inverse(input, upper=False, out=NULL) -\> Tensor

Computes the inverse of a symmetric positive-definite matrix \\A\\ using
its Cholesky factor \\u\\: returns matrix `inv`. The inverse is computed
using LAPACK routines `dpotri` and `spotri` (and the corresponding MAGMA
routines).

If `upper` is `FALSE`, \\u\\ is lower triangular such that the returned
tensor is

\$\$ inv = (uu^{{T}})^{{-1}} \$\$ If `upper` is `TRUE` or not provided,
\\u\\ is upper triangular such that the returned tensor is

\$\$ inv = (u^T u)^{{-1}} \$\$

## Examples

``` r
if (torch_is_installed()) {

if (FALSE) { # \dontrun{
a = torch_randn(c(3, 3))
a = torch_mm(a, a$t()) + 1e-05 * torch_eye(3) # make symmetric positive definite
u = torch_cholesky(a)
a
torch_cholesky_inverse(u)
a$inverse()
} # }
}
```
