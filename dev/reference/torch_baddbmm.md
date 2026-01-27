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
#>   0.7723 -0.0032 -0.5060  0.0580 -0.3655
#>  -0.7782 -1.0373  0.4520  2.0730  0.1241
#>  -0.1508 -0.2406 -0.4116  0.4093  4.8448
#> 
#> (2,.,.) = 
#>  -1.0769 -0.6013 -1.9306 -0.1158 -0.4632
#>  -0.5846 -1.4023  1.5442  1.9627  1.6822
#>  -1.1150  0.8493  2.2028 -1.2992  0.8188
#> 
#> (3,.,.) = 
#>   0.9969  0.7261  0.8166 -0.5807  1.7876
#>  -1.2625  0.6479  0.2152  1.2372  2.2773
#>   0.2890 -0.2876 -1.3165 -0.5174 -0.2789
#> 
#> (4,.,.) = 
#>   0.9833  1.1155  2.7802 -0.1949  1.6483
#>  -0.6492 -4.7536 -4.8869 -3.6554 -0.5498
#>  -2.6033  3.3532 -0.6157  1.9320 -0.4096
#> 
#> (5,.,.) = 
#>   0.7741  5.7400  1.3672  0.9497 -2.2133
#>  -3.2579 -1.2360  1.4858  1.3929 -0.5897
#>  -2.1390 -0.1063  0.5207  0.3372 -1.0091
#> 
#> (6,.,.) = 
#>  -0.0068  1.6652  3.8670  0.6424 -1.8262
#>  -0.1386  2.3834  5.4355 -3.2773 -0.3132
#>  -2.6162  5.0297 -5.0544  3.0285 -2.9989
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
