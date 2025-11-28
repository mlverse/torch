# Triangular_solve

Triangular_solve

## Usage

``` r
torch_triangular_solve(
  self,
  A,
  upper = TRUE,
  transpose = FALSE,
  unitriangular = FALSE
)
```

## Arguments

- self:

  (Tensor) multiple right-hand sides of size \\(\*, m, k)\\ where \\\*\\
  is zero of more batch dimensions (\\b\\)

- A:

  (Tensor) the input triangular coefficient matrix of size \\(\*, m,
  m)\\ where \\\*\\ is zero or more batch dimensions

- upper:

  (bool, optional) whether to solve the upper-triangular system of
  equations (default) or the lower-triangular system of equations.
  Default: `TRUE`.

- transpose:

  (bool, optional) whether \\A\\ should be transposed before being sent
  into the solver. Default: `FALSE`.

- unitriangular:

  (bool, optional) whether \\A\\ is unit triangular. If TRUE, the
  diagonal elements of \\A\\ are assumed to be 1 and not referenced from
  \\A\\. Default: `FALSE`.

## triangular_solve(input, A, upper=TRUE, transpose=False, unitriangular=False) -\> (Tensor, Tensor)

Solves a system of equations with a triangular coefficient matrix \\A\\
and multiple right-hand sides \\b\\.

In particular, solves \\AX = b\\ and assumes \\A\\ is upper-triangular
with the default keyword arguments.

`torch_triangular_solve(b, A)` can take in 2D inputs `b, A` or inputs
that are batches of 2D matrices. If the inputs are batches, then returns
batched outputs `X`

## Examples

``` r
if (torch_is_installed()) {

A = torch_randn(c(2, 2))$triu()
A
b = torch_randn(c(2, 3))
b
torch_triangular_solve(b, A)
}
#> [[1]]
#> torch_tensor
#> -1.0115  1.0974 -0.3162
#> -2.0546 -2.6187  1.7216
#> [ CPUFloatType{2,3} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.0416 -0.1884
#>  0.0000  0.6741
#> [ CPUFloatType{2,2} ]
#> 
```
