# Computes a solution to the least squares problem of a system of linear equations.

Letting be or , the **least squares problem** for a linear system with
is defined as

## Usage

``` r
linalg_lstsq(A, B, rcond = NULL, ..., driver = NULL)
```

## Arguments

- A:

  (Tensor): lhs tensor of shape `(*, m, n)` where `*` is zero or more
  batch dimensions.

- B:

  (Tensor): rhs tensor of shape `(*, m, k)` where `*` is zero or more
  batch dimensions.

- rcond:

  (float, optional): used to determine the effective rank of `A`. If
  `rcond = NULL`, `rcond` is set to the machine precision of the dtype
  of `A` times `max(m, n)`. Default: `NULL`.

- ...:

  currently unused.

- driver:

  (str, optional): name of the LAPACK/MAGMA method to be used. If
  `NULL`, `'gelsy'` is used for CPU inputs and `'gels'` for CUDA inputs.
  Default: `NULL`.

## Value

A list `(solution, residuals, rank, singular_values)`.

## Details

$$\underset{X \in \mathbb{K}^{n \times k}}{\min}\parallel AX - B\parallel_{F}$$X∈Kn×kmin​∥AX−B∥F​

where denotes the Frobenius norm. Supports inputs of float, double,
cfloat and cdouble dtypes.

Also supports batches of matrices, and if the inputs are batches of
matrices then the output has the same batch dimensions. `driver` chooses
the LAPACK/MAGMA function that will be used.

For CPU inputs the valid values are `'gels'`, `'gelsy'`, `'gelsd`,
`'gelss'`. For CUDA input, the only valid driver is `'gels'`, which
assumes that `A` is full-rank.

To choose the best driver on CPU consider:

- If `A` is well-conditioned (its [condition
  number](https://docs.pytorch.org/docs/master/linalg.html#torch.linalg.cond)
  is not too large), or you do not mind some precision loss.

- For a general matrix: `'gelsy'` (QR with pivoting) (default)

- If `A` is full-rank: `'gels'` (QR)

- If `A` is not well-conditioned.

- `'gelsd'` (tridiagonal reduction and SVD)

- But if you run into memory issues: `'gelss'` (full SVD).

See also the [full description of these
drivers](https://netlib.org/lapack/lug/node27.html)

`rcond` is used to determine the effective rank of the matrices in `A`
when `driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`). In this case,
if are the singular values of `A` in decreasing order, will be rounded
down to zero if . If `rcond = NULL` (default), `rcond` is set to the
machine precision of the dtype of `A`.

This function returns the solution to the problem and some extra
information in a list of four tensors
`(solution, residuals, rank, singular_values)`. For inputs `A`, `B` of
shape `(*, m, n)`, `(*, m, k)` respectively, it cointains

- `solution`: the least squares solution. It has shape `(*, n, k)`.

- `residuals`: the squared residuals of the solutions, that is, . It has
  shape equal to the batch dimensions of `A`. It is computed when
  `m > n` and every matrix in `A` is full-rank, otherwise, it is an
  empty tensor. If `A` is a batch of matrices and any matrix in the
  batch is not full rank, then an empty tensor is returned. This
  behavior may change in a future PyTorch release.

- `rank`: tensor of ranks of the matrices in `A`. It has shape equal to
  the batch dimensions of `A`. It is computed when `driver` is one of
  (`'gelsy'`, `'gelsd'`, `'gelss'`), otherwise it is an empty tensor.

- `singular_values`: tensor of singular values of the matrices in `A`.
  It has shape `(*, min(m, n))`. It is computed when `driver` is one of
  (`'gelsd'`, `'gelss'`), otherwise it is an empty tensor.

## Note

This function computes `X = A$pinverse() %*% B` in a faster and more
numerically stable way than performing the computations separately.

## Warning

The default value of `rcond` may change in a future PyTorch release. It
is therefore recommended to use a fixed value to avoid potential
breaking changes.

## See also

Other linalg:
[`linalg_cholesky()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky.md),
[`linalg_cholesky_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_cholesky_ex.md),
[`linalg_det()`](https://torch.mlverse.org/docs/dev/reference/linalg_det.md),
[`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md),
[`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md),
[`linalg_eigvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvals.md),
[`linalg_eigvalsh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigvalsh.md),
[`linalg_householder_product()`](https://torch.mlverse.org/docs/dev/reference/linalg_householder_product.md),
[`linalg_inv()`](https://torch.mlverse.org/docs/dev/reference/linalg_inv.md),
[`linalg_inv_ex()`](https://torch.mlverse.org/docs/dev/reference/linalg_inv_ex.md),
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
A <- torch_tensor(rbind(c(10, 2, 3), c(3, 10, 5), c(5, 6, 12)))$unsqueeze(1) # shape (1, 3, 3)
B <- torch_stack(list(
  rbind(c(2, 5, 1), c(3, 2, 1), c(5, 1, 9)),
  rbind(c(4, 2, 9), c(2, 0, 3), c(2, 5, 3))
), dim = 1) # shape (2, 3, 3)
X <- linalg_lstsq(A, B)$solution # A is broadcasted to shape (2, 3, 3)
}
```
