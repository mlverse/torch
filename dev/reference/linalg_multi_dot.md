# Efficiently multiplies two or more matrices

Efficiently multiplies two or more matrices by reordering the
multiplications so that the fewest arithmetic operations are performed.

## Usage

``` r
linalg_multi_dot(tensors)
```

## Arguments

- tensors:

  (`Sequence[Tensor]`): two or more tensors to multiply. The first and
  last tensors may be 1D or 2D. Every other tensor must be 2D.

## Details

Supports inputs of `float`, `double`, `cfloat` and `cdouble` dtypes.
This function does not support batched inputs.

Every tensor in `tensors` must be 2D, except for the first and last
which may be 1D. If the first tensor is a 1D vector of shape `(n,)` it
is treated as a row vector of shape `(1, n)`, similarly if the last
tensor is a 1D vector of shape `(n,)` it is treated as a column vector
of shape `(n, 1)`.

If the first and last tensors are matrices, the output will be a matrix.
However, if either is a 1D vector, then the output will be a 1D vector.

## Note

This function is implemented by chaining
[`torch_mm()`](https://torch.mlverse.org/docs/dev/reference/torch_mm.md)
calls after computing the optimal matrix multiplication order.

The cost of multiplying two matrices with shapes `(a, b)` and `(b, c)`
is `a * b * c`. Given matrices `A`, `B`, `C` with shapes `(10, 100)`,
`(100, 5)`, `(5, 50)` respectively, we can calculate the cost of
different multiplication orders as follows:

$$\begin{matrix}
{{cost}((AB)C)} & {= 10 \times 100 \times 5 + 10 \times 5 \times 50 = 7500\ {cost}(A(BC))} & {= 10 \times 100 \times 50 + 100 \times 5 \times 50 = 75000}
\end{matrix}$$cost((AB)C)​=10×100×5+10×5×50=7500 cost(A(BC))​=10×100×50+100×5×50=75000​

In this case, multiplying `A` and `B` first followed by `C` is 10 times
faster.

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
[`linalg_lstsq()`](https://torch.mlverse.org/docs/dev/reference/linalg_lstsq.md),
[`linalg_matrix_norm()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_norm.md),
[`linalg_matrix_power()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_power.md),
[`linalg_matrix_rank()`](https://torch.mlverse.org/docs/dev/reference/linalg_matrix_rank.md),
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

linalg_multi_dot(list(torch_tensor(c(1, 2)), torch_tensor(c(2, 3))))
}
#> torch_tensor
#> 8
#> [ CPUFloatType{} ]
```
