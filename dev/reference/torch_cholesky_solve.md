# Cholesky_solve

Cholesky_solve

## Usage

``` r
torch_cholesky_solve(self, input2, upper = FALSE)
```

## Arguments

- self:

  (Tensor) input matrix \\b\\ of size \\(\*, m, k)\\, where \\\*\\ is
  zero or more batch dimensions

- input2:

  (Tensor) input matrix \\u\\ of size \\(\*, m, m)\\, where \\\*\\ is
  zero of more batch dimensions composed of upper or lower triangular
  Cholesky factor

- upper:

  (bool, optional) whether to consider the Cholesky factor as a lower or
  upper triangular matrix. Default: `FALSE`.

## cholesky_solve(input, input2, upper=False, out=NULL) -\> Tensor

Solves a linear system of equations with a positive semidefinite matrix
to be inverted given its Cholesky factor matrix \\u\\.

If `upper` is `FALSE`, \\u\\ is and lower triangular and `c` is returned
such that:

\$\$ c = (u u^T)^{{-1}} b \$\$ If `upper` is `TRUE` or not provided,
\\u\\ is upper triangular and `c` is returned such that:

\$\$ c = (u^T u)^{{-1}} b \$\$ `torch_cholesky_solve(b, u)` can take in
2D inputs `b, u` or inputs that are batches of 2D matrices. If the
inputs are batches, then returns batched outputs `c`

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a = torch_mm(a, a$t()) # make symmetric positive definite
u = torch_cholesky(a)
a
b = torch_randn(c(3, 2))
b
torch_cholesky_solve(b, u)
torch_mm(a$inverse(), b)
}
#> torch_tensor
#>   34.0517   59.4075
#>  -20.5962  -36.9964
#>   88.7751  156.3272
#> [ CPUFloatType{3,2} ]
```
