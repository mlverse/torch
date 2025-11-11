# Sparse initialization

Fills the 2D input `Tensor` as a sparse matrix, where the non-zero
elements will be drawn from the normal distribution as described in
`Deep learning via Hessian-free optimization` - Martens, J. (2010).

## Usage

``` r
nn_init_sparse_(tensor, sparsity, std = 0.01)
```

## Arguments

- tensor:

  an n-dimensional `Tensor`

- sparsity:

  The fraction of elements in each column to be set to zero

- std:

  the standard deviation of the normal distribution used to generate the
  non-zero values

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
w <- torch_empty(3, 5)
nn_init_sparse_(w, sparsity = 0.1)
} # }
}
```
