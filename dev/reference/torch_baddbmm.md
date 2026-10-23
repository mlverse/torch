# Baddbmm

Baddbmm

## Usage

``` r
torch_baddbmm(self, batch1, batch2, out_dtype, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to be added

- batch1:

  (Tensor) the first batch of matrices to be multiplied

- batch2:

  (Tensor) the second batch of matrices to be multiplied

- out_dtype:

  (torch_dtype, optional) the output dtype

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
#> -1.8440  1.2104 -0.5493  2.7948 -1.3032
#>   0.1471 -1.5824  2.0627  2.0843 -2.9184
#>   0.3065  1.0966 -0.9102 -1.7023 -0.2864
#> 
#> (2,.,.) = 
#>  2.9475  0.0297 -6.7678  2.8091 -5.8769
#>  -0.2829 -1.4223 -5.9776  6.3847 -7.9571
#>  -1.9085 -2.0800 -0.0365  0.0245 -1.8927
#> 
#> (3,.,.) = 
#> -0.3779 -0.5146  3.4434 -4.8753 -0.1799
#>   0.6610 -2.3969  3.7187 -0.3328 -0.3768
#>   1.1331  0.7233 -0.8935 -1.1788 -1.9211
#> 
#> (4,.,.) = 
#> -4.1230 -1.7596  4.3497  1.8708  1.7812
#>  -1.8668 -2.3205  3.1217  2.8535  1.2627
#>   3.7414  1.6962  1.9452 -0.6624  1.6414
#> 
#> (5,.,.) = 
#> -0.4875  0.4368  0.1608 -3.1087 -3.2490
#>   0.2228 -0.0123 -0.3670 -1.4165 -1.3775
#>  -0.2027 -0.0251 -0.7539  1.2906  3.6480
#> 
#> (6,.,.) = 
#> -2.1389  4.6859 -0.3837  0.3339 -2.8928
#>   0.7450 -0.9066  4.2257 -3.1791 -5.4405
#>  -1.4448 -0.3860 -1.0232  0.1344  1.9718
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
