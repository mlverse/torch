# Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

Letting be or , the **eigenvalues** of a complex Hermitian or real
symmetric matrix are defined as the roots (counted with multiplicity) of
the polynomial `p` of degree `n` given by

## Usage

``` r
linalg_eigvalsh(A, UPLO = "L")
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch
  dimensions consisting of symmetric or Hermitian matrices.

- UPLO:

  ('L', 'U', optional): controls whether to use the upper or lower
  triangular part of `A` in the computations. Default: `'L'`.

## Value

A real-valued tensor cointaining the eigenvalues even when `A` is
complex. The eigenvalues are returned in ascending order.

## Details

$$p(\lambda) = \det(A - \lambda I_{n}){\qquad\lambda \in \mathbb{R}}$$p(λ)=det(A−λIn​)λ∈R

where is the `n`-dimensional identity matrix.

The eigenvalues of a real symmetric or complex Hermitian matrix are
always real. Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A` is a batch of matrices
then the output has the same batch dimensions. The eigenvalues are
returned in ascending order.

`A` is assumed to be Hermitian (resp. symmetric), but this is not
checked internally, instead:

- If `UPLO`\\ `= 'L'` (default), only the lower triangular part of the
  matrix is used in the computation.

- If `UPLO`\\ `= 'U'`, only the upper triangular part of the matrix is
  used.

## See also

- [`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md)
  computes the full eigenvalue decomposition.

Other linalg:
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
[`linalg_cholesky_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky_ex.md),
[`linalg_det()`](https://torch.mlverse.org/docs/dev/reference/linalg_det.md),
[`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md),
[`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md),
[`linalg_eigvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvals.md),
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
a <- torch_randn(2, 2)
linalg_eigvalsh(a)
}
#> torch_tensor
#> -0.5260
#>  0.7430
#> [ CPUFloatType{2} ]
```
