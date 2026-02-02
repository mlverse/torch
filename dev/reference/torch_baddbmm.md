# Baddbmm

Baddbmm

## Usage

``` r
torch_baddbmm(self, batch1, batch2, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to be added

- batch1:

  (Tensor) the first batch of matrices to be multiplied

- batch2:

  (Tensor) the second batch of matrices to be multiplied

- beta:

  (Number, optional) multiplier for `input` (\\\beta\\)

- alpha:

  (Number, optional) multiplier for \\\mbox{batch1} \mathbin{@}
  \mbox{batch2}\\ (\\\alpha\\)

## baddbmm(input, batch1, batch2, \*, beta=1, alpha=1, out=NULL) -\> Tensor

Performs a batch matrix-matrix product of matrices in `batch1` and
`batch2`. `input` is added to the final result.

`batch1` and `batch2` must be 3-D tensors each containing the same
number of matrices.

If `batch1` is a \\(b \times n \times m)\\ tensor, `batch2` is a \\(b
\times m \times p)\\ tensor, then `input` must be broadcastable with a
\\(b \times n \times p)\\ tensor and `out` will be a \\(b \times n
\times p)\\ tensor. Both `alpha` and `beta` mean the same as the scaling
factors used in `torch_addbmm`.

\$\$ \mbox{out}\_i = \beta\\ \mbox{input}\_i + \alpha\\
(\mbox{batch1}\_i \mathbin{@} \mbox{batch2}\_i) \$\$ For inputs of type
`FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be
real numbers, otherwise they should be integers.

## Examples

``` r
if (torch_is_installed()) {

M = torch_randn(c(10, 3, 5))
batch1 = torch_randn(c(10, 3, 4))
batch2 = torch_randn(c(10, 4, 5))
torch_baddbmm(M, batch1, batch2)
}
#> torch_tensor
#> (1,.,.) = 
#>   5.4965  3.1711 -5.1700 -5.7073 -0.3017
#>  -2.4449 -1.7671  2.6752  0.0412  0.7662
#>  -0.1733 -2.2723  0.7899 -0.6253 -0.8835
#> 
#> (2,.,.) = 
#>   0.9054  0.3918 -1.5563 -0.5508 -0.2275
#>  -2.3430 -2.9237  2.2118  0.4519  1.4373
#>   3.0306  0.0237 -1.7022  2.0918 -1.9170
#> 
#> (3,.,.) = 
#>  -0.5751 -0.1596 -0.4127 -0.6751 -0.8338
#>   0.8944  0.7495 -0.7671 -7.1824  0.6302
#>   0.1728  0.1598  1.7942 -8.4603 -4.3257
#> 
#> (4,.,.) = 
#>   0.2051 -4.1667 -0.4968  1.7877  0.2434
#>   1.8425 -0.5174 -0.1250  2.6946  0.1188
#>  -2.7167  1.2612 -3.2872  1.9201  1.0962
#> 
#> (5,.,.) = 
#>  -2.9336 -0.7681  1.1801  2.0574 -0.1178
#>   3.9441 -2.3882 -4.3597 -1.3290  1.5027
#>  -1.9726  1.0937  2.2706  0.8549  1.9134
#> 
#> (6,.,.) = 
#>  -0.9897  1.7891 -1.1165  1.0353  1.2869
#>  -1.2473 -2.1507  3.1299  0.8675  0.5725
#>   2.7330  0.7798  0.6324 -1.4968 -0.2508
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
