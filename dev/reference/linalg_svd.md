# Computes the singular value decomposition (SVD) of a matrix.

Letting be or , the **full SVD** of a matrix , if `k = min(m,n)`, is
defined as

## Usage

``` r
linalg_svd(A, full_matrices = TRUE)
```

## Arguments

- A:

  (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch
  dimensions.

- full_matrices:

  (bool, optional): controls whether to compute the full or reduced SVD,
  and consequently, the shape of the returned tensors `U` and `V`.
  Default: `TRUE`.

## Value

A list `(U, S, V)` which corresponds to , , above. `S` will always be
real-valued, even when `A` is complex. It will also be ordered in
descending order. `U` and `V` will have the same dtype as `A`. The left
/ right singular vectors will be given by the columns of `U` and the
rows of `V` respectively.

## Details

$$A = U{diag}(S)V^{H}{\qquad U \in \mathbb{K}^{m \times m},S \in \mathbb{R}^{k},V \in \mathbb{K}^{n \times n}}$$A=Udiag(S)VHU∈Km×m,S∈Rk,V∈Kn×n

where , is the conjugate transpose when is complex, and the transpose
when is real-valued.

The matrices , (and thus ) are orthogonal in the real case, and unitary
in the complex case. When `m > n` (resp. `m < n`) we can drop the last
`m - n` (resp. `n - m`) columns of `U` (resp. `V`) to form the **reduced
SVD**:

$$A = U{diag}(S)V^{H}{\qquad U \in \mathbb{K}^{m \times k},S \in \mathbb{R}^{k},V \in \mathbb{K}^{k \times n}}$$A=Udiag(S)VHU∈Km×k,S∈Rk,V∈Kk×n

where .

In this case, and also have orthonormal columns. Supports input of
float, double, cfloat and cdouble dtypes.

Also supports batches of matrices, and if `A` is a batch of matrices
then the output has the same batch dimensions.

The returned decomposition is a named tuple `(U, S, V)` which
corresponds to , , above.

The singular values are returned in descending order. The parameter
`full_matrices` chooses between the full (default) and reduced SVD.

## Note

When `full_matrices=TRUE`, the gradients with respect to
`U[..., :, min(m, n):]` and `Vh[..., min(m, n):, :]` will be ignored, as
those vectors can be arbitrary bases of the corresponding subspaces.

## Warnings

The returned tensors `U` and `V` are not unique, nor are they continuous
with respect to `A`. Due to this lack of uniqueness, different hardware
and software may compute different singular vectors. This non-uniqueness
is caused by the fact that multiplying any pair of singular vectors by
`-1` in the real case or by in the complex case produces another two
valid singular vectors of the matrix. This non-uniqueness problem is
even worse when the matrix has repeated singular values. In this case,
one may multiply the associated singular vectors of `U` and `V` spanning
the subspace by a rotation matrix and the resulting vectors will span
the same subspace.

Gradients computed using `U` or `V` will only be finite when `A` does
not have zero as a singular value or repeated singular values.
Furthermore, if the distance between any two singular values is close to
zero, the gradient will be numerically unstable, as it depends on the
singular values through the computation of . The gradient will also be
numerically unstable when `A` has small singular values, as it also
depends on the computaiton of .

## See also

- [`linalg_svdvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_svdvals.md)
  computes only the singular values. Unlike `linalg_svd()`, the
  gradients of
  [`linalg_svdvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_svdvals.md)
  are always numerically stable.

- [`linalg_eig()`](https://torch.mlverse.org/docs/dev/reference/linalg_eig.md)
  for a function that computes another type of spectral decomposition of
  a matrix. The eigendecomposition works just on on square matrices.

- [`linalg_eigh()`](https://torch.mlverse.org/docs/dev/reference/linalg_eigh.md)
  for a (faster) function that computes the eigenvalue decomposition for
  Hermitian and symmetric matrices.

- [`linalg_qr()`](https://torch.mlverse.org/docs/dev/reference/linalg_qr.md)
  for another (much faster) decomposition that works on general
  matrices.

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
[`linalg_svdvals()`](https://torch.mlverse.org/docs/dev/reference/linalg_svdvals.md),
[`linalg_tensorinv()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorinv.md),
[`linalg_tensorsolve()`](https://torch.mlverse.org/docs/dev/reference/linalg_tensorsolve.md),
[`linalg_vector_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_vector_norm.md)

## Examples

``` r
if (torch_is_installed()) {

a <- torch_randn(5, 3)
linalg_svd(a, full_matrices = FALSE)
}
#> [[1]]
#> torch_tensor
#>  0.0479 -0.4428  0.5105
#> -0.3284  0.3795  0.4224
#>  0.3095 -0.2562 -0.6186
#> -0.7365  0.2012 -0.4222
#> -0.5017 -0.7442  0.0104
#> [ CPUFloatType{5,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  2.1883
#>  1.9491
#>  0.5115
#> [ CPUFloatType{3} ]
#> 
#> [[3]]
#> torch_tensor
#>  0.2086  0.9086  0.3619
#>  0.1360  0.3395 -0.9307
#>  0.9685 -0.2434  0.0528
#> [ CPUFloatType{3,3} ]
#> 
```
