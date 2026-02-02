# Addr

Addr

## Usage

``` r
torch_addr(self, vec1, vec2, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) matrix to be added

- vec1:

  (Tensor) the first vector of the outer product

- vec2:

  (Tensor) the second vector of the outer product

- beta:

  (Number, optional) multiplier for `input` (\\\beta\\)

- alpha:

  (Number, optional) multiplier for \\\mbox{vec1} \otimes \mbox{vec2}\\
  (\\\alpha\\)

## addr(input, vec1, vec2, \*, beta=1, alpha=1, out=NULL) -\> Tensor

Performs the outer-product of vectors `vec1` and `vec2` and adds it to
the matrix `input`.

Optional values `beta` and `alpha` are scaling factors on the outer
product between `vec1` and `vec2` and the added matrix `input`
respectively.

\$\$ \mbox{out} = \beta\\ \mbox{input} + \alpha\\ (\mbox{vec1} \otimes
\mbox{vec2}) \$\$ If `vec1` is a vector of size `n` and `vec2` is a
vector of size `m`, then `input` must be broadcastable with a matrix of
size \\(n \times m)\\ and `out` will be a matrix of size \\(n \times
m)\\.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and
`alpha` must be real numbers, otherwise they should be integers

## Examples

``` r
if (torch_is_installed()) {

vec1 = torch_arange(1, 3)
vec2 = torch_arange(1, 2)
M = torch_zeros(c(3, 2))
torch_addr(M, vec1, vec2)
}
#> torch_tensor
#>  1  2
#>  2  4
#>  3  6
#> [ CPUFloatType{3,2} ]
```
