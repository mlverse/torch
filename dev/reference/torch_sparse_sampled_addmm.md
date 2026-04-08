# Sparse_sampled_addmm

Sparse_sampled_addmm

## Usage

``` r
torch_sparse_sampled_addmm(self, mat1, mat2, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) a sparse CSR matrix of size `(m, n)` to be added and used to
  compute the sampled matrix multiplication

- mat1:

  (Tensor) a dense matrix of size `(m, k)` to be multiplied

- mat2:

  (Tensor) a dense matrix of size `(k, n)` to be multiplied

- beta:

  (Number, optional) multiplier for `self` (\\\beta\\)

- alpha:

  (Number, optional) multiplier for \\mat1 @ mat2\\ (\\\alpha\\)

## sparse_sampled_addmm(self, mat1, mat2, \*, beta=1, alpha=1) -\> Tensor

Performs a matrix multiplication of the dense matrices `mat1` and `mat2`
at the locations specified by the sparsity pattern of `self`. The matrix
`self` is added to the final result.

Mathematically, this performs the following operation:

\$\$ \mbox{out} = \alpha\\ (\mbox{mat1} \mathbin{@} \mbox{mat2}) \*
\mbox{spy}(\mbox{self}) + \beta\\ \mbox{self} \$\$

where \\\mbox{spy}(\mbox{self})\\ is the sparsity pattern matrix of
`self`, `alpha` and `beta` are the scaling factors.
\\\mbox{spy}(\mbox{self})\\ has value 1 at the positions where `self`
has non-zero values, and 0 elsewhere.

`self` must be a sparse CSR tensor. `mat1` and `mat2` must be dense
tensors.

## Examples

``` r
if (torch_is_installed()) {

# Create a sparse CSR mask from a COO tensor
i <- torch_tensor(matrix(c(1, 2, 2, 1, 2, 3), nrow = 2, byrow = TRUE),
  dtype = torch_long())
v <- torch_tensor(c(1, 2, 3), dtype = torch_float32())
sparse_mask <- torch_sparse_coo_tensor(i, v, c(2, 3))$to_sparse_csr()

mat1 <- torch_randn(c(2, 3))
mat2 <- torch_randn(c(3, 3))

result <- torch_sparse_sampled_addmm(sparse_mask, mat1, mat2)
result$to_dense()

}
```
