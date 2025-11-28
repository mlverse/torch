# Computes the inverse of a square matrix if it exists.

Throws a `runtime_error` if the matrix is not invertible.

## Usage

``` r
linalg_inv(A)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch
  dimensions consisting of invertible matrices.

## Details

Letting be or , for a matrix , its **inverse matrix** (if it exists) is
defined as

$$A^{- 1}A = AA^{- 1} = I_{n}$$A−1A=AA−1=In​ where is the `n`-dimensional
identity matrix.

The inverse matrix exists if and only if is invertible. In this case,
the inverse is unique. Supports input of float, double, cfloat and
cdouble dtypes. Also supports batches of matrices, and if `A` is a batch
of matrices then the output has the same batch dimensions.

Consider using
[`linalg_solve()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve.md)
if possible for multiplying a matrix on the left by the inverse, as
`linalg_solve(A, B) == A$inv() %*% B` It is always prefered to use
[`linalg_solve()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve.md)
when possible, as it is faster and more numerically stable than
computing the inverse explicitly.

## See also

[`linalg_pinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_pinv.md)
computes the pseudoinverse (Moore-Penrose inverse) of matrices of any
shape.
[`linalg_solve()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve.md)
computes `A$inv() %*% B` with a numerically stable algorithm.

Other linalg:
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
[`linalg_cholesky_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky_ex.md),
[`linalg_det()`](https://torch.mlverse.org/docs/dev/reference/linalg_det.md),
[`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md),
[`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md),
[`linalg_eigvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvals.md),
[`linalg_eigvalsh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvalsh.md),
[`linalg_householder_product()`](https://torch.mlverse.org/docs/dev/reference/linalg_householder_product.md),
[`linalg_inv_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_inv_ex.md),
[`linalg_lstsq()`](https://torch.mlverse.org/docs/dev/reference/linalg_lstsq.md),
[`linalg_matrix_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_norm.md),
[`linalg_matrix_power()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_power.md),
[`linalg_matrix_rank()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_rank.md),
[`linalg_multi_dot()`](https://torch.mlverse.org/docs/dev/reference/linalg_multi_dot.md),
[`linalg_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_norm.md),
[`linalg_pinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_pinv.md),
[`linalg_qr()`](https://torch.mlverse.org/docs/dev/reference/linalg_qr.md),
[`linalg_slogdet()`](https://torch.mlverse.org/docs/dev/reference/linalg_slogdet.md),
[`linalg_solve()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve.md),
[`linalg_solve_triangular()`](https://torch.mlverse.org/docs/dev/reference/linalg_solve_triangular.md),
[`linalg_svd()`](https://torch.mlverse.org/docs/dev/reference/linalg_svd.md),
[`linalg_svdvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_svdvals.md),
[`linalg_tensorinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorinv.md),
[`linalg_tensorsolve()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorsolve.md),
[`linalg_vector_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_vector_norm.md)

## Examples

``` r
if (torch_is_installed()) {
A <- torch_randn(4, 4)
linalg_inv(A)
}
#> torch_tensor
#> -2.3284 -1.3506  0.0209  0.2466
#>  0.1902 -0.4976 -0.4296  0.5954
#>  4.8657  1.5865  0.8598 -0.2536
#> -1.3963 -0.7542 -0.3766 -0.4091
#> [ CPUFloatType{4,4} ]
```
