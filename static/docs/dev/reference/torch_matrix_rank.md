# Matrix_rank

Matrix_rank

## Arguments

- self:

  (Tensor) the input 2-D tensor

- tol:

  (float, optional) the tolerance value. Default: `NULL`

- symmetric:

  (bool, optional) indicates whether `input` is symmetric. Default:
  `FALSE`

## matrix_rank(input, tol=NULL, symmetric=False) -\> Tensor

Returns the numerical rank of a 2-D tensor. The method to compute the
matrix rank is done using SVD by default. If `symmetric` is `TRUE`, then
`input` is assumed to be symmetric, and the computation of the rank is
done by obtaining the eigenvalues.

`tol` is the threshold below which the singular values (or the
eigenvalues when `symmetric` is `TRUE`) are considered to be 0. If `tol`
is not specified, `tol` is set to `S.max() * max(S.size()) * eps` where
`S` is the singular values (or the eigenvalues when `symmetric` is
`TRUE`), and `eps` is the epsilon value for the datatype of `input`.
