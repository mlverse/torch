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
#>   4.3062 -1.5423  1.8580 -2.0842  1.0167
#>  -3.0711 -1.2423  2.2234  1.0221  0.7033
#>  -3.8092  1.0742 -2.2874  0.8563 -1.5219
#> 
#> (2,.,.) = 
#>  -0.4796  2.0828 -1.6737 -1.6285 -2.7214
#>   0.4082 -2.4441 -2.0911  2.6761 -4.8673
#>   0.4884  3.4539 -1.5421 -2.1923 -1.5772
#> 
#> (3,.,.) = 
#>  -1.9633  1.4375 -1.3572 -0.9815  0.5351
#>  -0.5905  0.2635  2.5859 -0.8742  0.4257
#>  -1.2046 -1.3401 -1.2376 -2.0975  1.5077
#> 
#> (4,.,.) = 
#>   0.3860  3.7764 -1.5785 -2.6312  0.4589
#>   0.0932  7.2830 -0.7196 -2.2517  1.0745
#>  -2.8887 -2.7930 -2.8603  0.1514  0.0774
#> 
#> (5,.,.) = 
#>   4.0702  4.8749 -0.1398 -0.0054  2.1467
#>  -0.4884 -1.3378 -0.2727 -2.0809  2.6286
#>  -0.9931  5.3078  0.0497 -2.2653  0.5712
#> 
#> (6,.,.) = 
#>   0.5361 -0.3993 -0.5828 -0.0577  1.9910
#>   1.8247  1.5525  0.5062 -1.3688  0.7088
#>   2.5519 -5.9783 -6.9031 -3.0021  5.0127
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
