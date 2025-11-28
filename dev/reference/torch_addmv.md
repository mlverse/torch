# Addmv

Addmv

## Usage

``` r
torch_addmv(self, mat, vec, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) vector to be added

- mat:

  (Tensor) matrix to be multiplied

- vec:

  (Tensor) vector to be multiplied

- beta:

  (Number, optional) multiplier for `input` (\\\beta\\)

- alpha:

  (Number, optional) multiplier for \\mat @ vec\\ (\\\alpha\\)

## addmv(input, mat, vec, \*, beta=1, alpha=1, out=NULL) -\> Tensor

Performs a matrix-vector product of the matrix `mat` and the vector
`vec`. The vector `input` is added to the final result.

If `mat` is a \\(n \times m)\\ tensor, `vec` is a 1-D tensor of size
`m`, then `input` must be broadcastable with a 1-D tensor of size `n`
and `out` will be 1-D tensor of size `n`.

`alpha` and `beta` are scaling factors on matrix-vector product between
`mat` and `vec` and the added tensor `input` respectively.

\$\$ \mbox{out} = \beta\\ \mbox{input} + \alpha\\ (\mbox{mat}
\mathbin{@} \mbox{vec}) \$\$ For inputs of type `FloatTensor` or
`DoubleTensor`, arguments `beta` and `alpha` must be real numbers,
otherwise they should be integers

## Examples

``` r
if (torch_is_installed()) {

M = torch_randn(c(2))
mat = torch_randn(c(2, 3))
vec = torch_randn(c(3))
torch_addmv(M, mat, vec)
}
#> torch_tensor
#> -3.9066
#>  0.3663
#> [ CPUFloatType{2} ]
```
