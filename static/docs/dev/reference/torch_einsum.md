# Einsum

Einsum

## Usage

``` r
torch_einsum(equation, tensors, path = NULL)
```

## Arguments

- equation:

  (string) The equation is given in terms of lower case letters
  (indices) to be associated with each dimension of the operands and
  result. The left hand side lists the operands dimensions, separated by
  commas. There should be one index letter per tensor dimension. The
  right hand side follows after `->` and gives the indices for the
  output. If the `->` and right hand side are omitted, it implicitly
  defined as the alphabetically sorted list of all indices appearing
  exactly once in the left hand side. The indices not apprearing in the
  output are summed over after multiplying the operands entries. If an
  index appears several times for the same operand, a diagonal is taken.
  Ellipses `...` represent a fixed number of dimensions. If the right
  hand side is inferred, the ellipsis dimensions are at the beginning of
  the output.

- tensors:

  (Tensor) The operands to compute the Einstein sum of.

- path:

  (int) This function uses
  [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/) to
  speed up computation or to consume less memory by optimizing
  contraction order. This optimization occurs when there are at least
  three inputs, since the order does not matter otherwise. Note that
  finding *the* optimal path is an NP-hard problem, thus, `opt_einsum`
  relies on different heuristics to achieve near-optimal results. If
  `opt_einsum` is not available, the default order is to contract from
  left to right. The path argument is used to changed that default, but
  it should only be set by advanced users.

## einsum(equation, \*operands) -\> Tensor

This function provides a way of computing multilinear expressions (i.e.
sums of products) using the Einstein summation convention.

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(5))
y = torch_randn(c(4))
torch_einsum('i,j->ij', list(x, y))  # outer product
A = torch_randn(c(3,5,4))
l = torch_randn(c(2,5))
r = torch_randn(c(2,4))
torch_einsum('bn,anm,bm->ba', list(l, A, r)) # compare torch_nn$functional$bilinear
As = torch_randn(c(3,2,5))
Bs = torch_randn(c(3,5,4))
torch_einsum('bij,bjk->bik', list(As, Bs)) # batch matrix multiplication
A = torch_randn(c(3, 3))
torch_einsum('ii->i', list(A)) # diagonal
A = torch_randn(c(4, 3, 3))
torch_einsum('...ii->...i', list(A)) # batch diagonal
A = torch_randn(c(2, 3, 4, 5))
torch_einsum('...ij->...ji', list(A))$shape # batch permute

}
#> [1] 2 3 5 4
```
