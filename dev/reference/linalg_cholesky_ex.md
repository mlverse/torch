# Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

This function skips the (slow) error checking and error message
construction of
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
instead directly returning the LAPACK error codes as part of a named
tuple `(L, info)`. This makes this function a faster way to check if a
matrix is positive-definite, and it provides an opportunity to handle
decomposition errors more gracefully or performantly than
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md)
does. Supports input of float, double, cfloat and cdouble dtypes. Also
supports batches of matrices, and if `A` is a batch of matrices then the
output has the same batch dimensions. If `A` is not a Hermitian
positive-definite matrix, or if it's a batch of matrices and one or more
of them is not a Hermitian positive-definite matrix, then `info` stores
a positive integer for the corresponding matrix. The positive integer
indicates the order of the leading minor that is not positive-definite,
and the decomposition could not be completed. `info` filled with zeros
indicates that the decomposition was successful. If `check_errors=TRUE`
and `info` contains positive integers, then a RuntimeError is thrown.

## Usage

``` r
linalg_cholesky_ex(A, check_errors = FALSE)
```

## Arguments

- A:

  (Tensor): the Hermitian `n \times n` matrix or the batch of such
  matrices of size `(*, n, n)` where `*` is one or more batch
  dimensions.

- check_errors:

  (bool, optional): controls whether to check the content of `infos`.
  Default: `FALSE`.

## Note

If `A` is on a CUDA device, this function may synchronize that device
with the CPU.

This function is "experimental" and it may change in a future PyTorch
release.

## See also

[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md)
is a NumPy compatible variant that always checks for errors.

Other linalg:
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
[`linalg_det()`](https://torch.mlverse.org/docs/dev/reference/linalg_det.md),
[`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md),
[`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md),
[`linalg_eigvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvals.md),
[`linalg_eigvalsh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvalsh.md),
[`linalg_householder_product()`](https://torch.mlverse.org/docs/dev/reference/linalg_householder_product.md),
[`linalg_inv()`](https://torch.mlverse.org/docs/dev/reference/linalg_inv.md),
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
A <- torch_randn(2, 2)
out <- linalg_cholesky_ex(A)
out
}
#> $L
#> torch_tensor
#> -0.4271  0.0000
#>  0.6837 -0.4999
#> [ CPUFloatType{2,2} ]
#> 
#> $info
#> torch_tensor
#> 1
#> [ CPUIntType{} ]
#> 
```
