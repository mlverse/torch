# Lu_solve

Lu_solve

## Usage

``` r
torch_lu_solve(self, LU_data, LU_pivots)
```

## Arguments

- self:

  (Tensor) the RHS tensor of size \\(\*, m, k)\\, where \\\*\\ is zero
  or more batch dimensions.

- LU_data:

  (Tensor) the pivoted LU factorization of A from `torch_lu` of size
  \\(\*, m, m)\\, where \\\*\\ is zero or more batch dimensions.

- LU_pivots:

  (IntTensor) the pivots of the LU factorization from `torch_lu` of size
  \\(\*, m)\\, where \\\*\\ is zero or more batch dimensions. The batch
  dimensions of `LU_pivots` must be equal to the batch dimensions of
  `LU_data`.

## lu_solve(input, LU_data, LU_pivots, out=NULL) -\> Tensor

Returns the LU solve of the linear system \\Ax = b\\ using the partially
pivoted LU factorization of A from `torch_lu`.

## Examples

``` r
if (torch_is_installed()) {
A = torch_randn(c(2, 3, 3))
b = torch_randn(c(2, 3, 1))
out = torch_lu(A)
x = torch_lu_solve(b, out[[1]], out[[2]])
torch_norm(torch_bmm(A, x) - b)
}
#> torch_tensor
#> 4.28667e-07
#> [ CPUFloatType{} ]
```
