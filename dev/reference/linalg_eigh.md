# Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.

Letting be or , the **eigenvalue decomposition** of a complex Hermitian
or real symmetric matrix is defined as

## Usage

``` r
linalg_eigh(A, UPLO = "L")
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch
  dimensions consisting of symmetric or Hermitian matrices.

- UPLO:

  ('L', 'U', optional): controls whether to use the upper or lower
  triangular part of `A` in the computations. Default: `'L'`.

## Value

A list `(eigenvalues, eigenvectors)` which corresponds to and above.
`eigenvalues` will always be real-valued, even when `A` is complex.

It will also be ordered in ascending order. `eigenvectors` will have the
same dtype as `A` and will contain the eigenvectors as its columns.

## Details

$$A = Q{diag}(\Lambda)Q^{H}{\qquad Q \in \mathbb{K}^{n \times n},\Lambda \in \mathbb{R}^{n}}$$A=Qdiag(Λ)QHQ∈Kn×n,Λ∈Rn

where is the conjugate transpose when is complex, and the transpose when
is real-valued. is orthogonal in the real case and unitary in the
complex case.

Supports input of float, double, cfloat and cdouble dtypes. Also
supports batches of matrices, and if `A` is a batch of matrices then the
output has the same batch dimensions.

`A` is assumed to be Hermitian (resp. symmetric), but this is not
checked internally, instead:

- If `UPLO`\\ `= 'L'` (default), only the lower triangular part of the
  matrix is used in the computation.

- If `UPLO`\\ `= 'U'`, only the upper triangular part of the matrix is
  used. The eigenvalues are returned in ascending order.

## Note

The eigenvalues of real symmetric or complex Hermitian matrices are
always real.

## Warning

- The eigenvectors of a symmetric matrix are not unique, nor are they
  continuous with respect to `A`. Due to this lack of uniqueness,
  different hardware and software may compute different eigenvectors.
  This non-uniqueness is caused by the fact that multiplying an
  eigenvector by `-1` in the real case or by in the complex case
  produces another set of valid eigenvectors of the matrix. This
  non-uniqueness problem is even worse when the matrix has repeated
  eigenvalues. In this case, one may multiply the associated
  eigenvectors spanning the subspace by a rotation matrix and the
  resulting eigenvectors will be valid eigenvectors.

- Gradients computed using the `eigenvectors` tensor will only be finite
  when `A` has unique eigenvalues. Furthermore, if the distance between
  any two eigvalues is close to zero, the gradient will be numerically
  unstable, as it depends on the eigenvalues through the computation of
  .

## See also

- [`linalg_eigvalsh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvalsh.md)
  computes only the eigenvalues values of a Hermitian matrix. Unlike
  `linalg_eigh()`, the gradients of
  [`linalg_eigvalsh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvalsh.md)
  are always numerically stable.

- [`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md)
  for a different decomposition of a Hermitian matrix. The Cholesky
  decomposition gives less information about the matrix but is much
  faster to compute than the eigenvalue decomposition.

- [`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md)
  for a (slower) function that computes the eigenvalue decomposition of
  a not necessarily Hermitian square matrix.

- [`linalg_svd()`](https://torch.mlverse.org/docs/dev/reference/linalg_svd.md)
  for a (slower) function that computes the more general SVD
  decomposition of matrices of any shape.

- [`linalg_qr()`](https://torch.mlverse.org/docs/dev/reference/linalg_qr.md)
  for another (much faster) decomposition that works on general
  matrices.

Other linalg:
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
[`linalg_cholesky_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky_ex.md),
[`linalg_det()`](https://torch.mlverse.org/docs/dev/reference/linalg_det.md),
[`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md),
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
a <- torch_randn(2, 2)
linalg_eigh(a)
}
#> [[1]]
#> torch_tensor
#> -0.3424
#>  0.6204
#> [ CPUFloatType{2} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.1446 -0.9895
#> -0.9895  0.1446
#> [ CPUFloatType{2,2} ]
#> 
```
