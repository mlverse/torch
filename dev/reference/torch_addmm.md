# Addmm

Addmm

## Usage

``` r
torch_addmm(self, mat1, mat2, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) matrix to be added

- mat1:

  (Tensor) the first matrix to be multiplied

- mat2:

  (Tensor) the second matrix to be multiplied

- beta:

  (Number, optional) multiplier for `input` (\\\beta\\)

- alpha:

  (Number, optional) multiplier for \\mat1 @ mat2\\ (\\\alpha\\)

## addmm(input, mat1, mat2, \*, beta=1, alpha=1, out=NULL) -\> Tensor

Performs a matrix multiplication of the matrices `mat1` and `mat2`. The
matrix `input` is added to the final result.

If `mat1` is a \\(n \times m)\\ tensor, `mat2` is a \\(m \times p)\\
tensor, then `input` must be broadcastable with a \\(n \times p)\\
tensor and `out` will be a \\(n \times p)\\ tensor.

`alpha` and `beta` are scaling factors on matrix-vector product between
`mat1` and `mat2` and the added matrix `input` respectively.

\$\$ \mbox{out} = \beta\\ \mbox{input} + \alpha\\ (\mbox{mat1}\_i
\mathbin{@} \mbox{mat2}\_i) \$\$ For inputs of type `FloatTensor` or
`DoubleTensor`, arguments `beta` and `alpha` must be real numbers,
otherwise they should be integers.

## Examples

``` r
if (torch_is_installed()) {

M = torch_randn(c(2, 3))
mat1 = torch_randn(c(2, 3))
mat2 = torch_randn(c(3, 3))
torch_addmm(M, mat1, mat2)
}
#> torch_tensor
#>  3.8758  1.6139 -2.4089
#>  2.2462 -0.8908 -1.0654
#> [ CPUFloatType{2,3} ]
```
