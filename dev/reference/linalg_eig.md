# Computes the eigenvalue decomposition of a square matrix if it exists.

Letting be or , the **eigenvalue decomposition** of a square matrix (if
it exists) is defined as

## Usage

``` r
linalg_eig(A)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch
  dimensions consisting of diagonalizable matrices.

## Value

A list `(eigenvalues, eigenvectors)` which corresponds to and above.
`eigenvalues` and `eigenvectors` will always be complex-valued, even
when `A` is real. The eigenvectors will be given by the columns of
`eigenvectors`.

## Details

$$A = V{diag}(\Lambda)V^{- 1}{\qquad V \in \mathbb{C}^{n \times n},\Lambda \in \mathbb{C}^{n}}$$A=Vdiag(Λ)V−1V∈Cn×n,Λ∈Cn

This decomposition exists if and only if is `diagonalizable`\_. This is
the case when all its eigenvalues are different. Supports input of
float, double, cfloat and cdouble dtypes. Also supports batches of
matrices, and if `A` is a batch of matrices then the output has the same
batch dimensions.

## Note

The eigenvalues and eigenvectors of a real matrix may be complex.

## Warning

- This function assumes that `A` is `diagonalizable`\_ (for example,
  when all the eigenvalues are different). If it is not diagonalizable,
  the returned eigenvalues will be correct but .

- The eigenvectors of a matrix are not unique, nor are they continuous
  with respect to `A`. Due to this lack of uniqueness, different
  hardware and software may compute different eigenvectors. This
  non-uniqueness is caused by the fact that multiplying an eigenvector
  by a non-zero number produces another set of valid eigenvectors of the
  matrix. In this implmentation, the returned eigenvectors are
  normalized to have norm `1` and largest real component.

- Gradients computed using `V` will only be finite when `A` does not
  have repeated eigenvalues. Furthermore, if the distance between any
  two eigenvalues is close to zero, the gradient will be numerically
  unstable, as it depends on the eigenvalues through the computation of
  .

## See also

- [`linalg_eigvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvals.md)
  computes only the eigenvalues. Unlike `linalg_eig()`, the gradients of
  [`linalg_eigvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvals.md)
  are always numerically stable.

- [`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md)
  for a (faster) function that computes the eigenvalue decomposition for
  Hermitian and symmetric matrices.

- [`linalg_svd()`](https://torch.mlverse.org/docs/dev/reference/linalg_svd.md)
  for a function that computes another type of spectral decomposition
  that works on matrices of any shape.

- [`linalg_qr()`](https://torch.mlverse.org/docs/dev/reference/linalg_qr.md)
  for another (much faster) decomposition that works on matrices of any
  shape.

Other linalg:
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
[`linalg_cholesky_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky_ex.md),
[`linalg_det()`](https://torch.mlverse.org/docs/dev/reference/linalg_det.md),
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
a <- torch_randn(2, 2)
wv <- linalg_eig(a)
}
```
