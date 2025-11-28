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
#>  -5.2007  0.0807  3.0940 -1.3121  2.5306
#>   0.2644 -1.2034 -3.5998  5.4335 -0.2692
#>  -0.3976 -2.9182 -0.8564  4.3631 -2.4356
#> 
#> (2,.,.) = 
#>   0.3607  0.9992  3.1091 -1.1227  1.6882
#>   1.5847  4.5565 -4.8489 -5.2819 -6.7872
#>  -2.1788  0.9322 -0.0206 -2.1114  1.0077
#> 
#> (3,.,.) = 
#>  -0.9956  1.2538 -0.1740  0.4568 -1.5000
#>   1.1305  0.5745  0.1917 -1.0205  0.6596
#>  -2.2767  0.1652 -0.3900 -1.4924  1.5098
#> 
#> (4,.,.) = 
#>  -1.5876  1.5219 -1.7912  4.3624  3.1886
#>   4.4580  0.4052  0.2761  0.9690  2.0056
#>  -2.3469  0.2802  2.6563  0.1204 -1.4903
#> 
#> (5,.,.) = 
#>   4.2489 -1.0415  3.2728  1.4116  3.9124
#>   6.9167 -0.0519 -1.7989  0.1661  3.2426
#>  -6.8922 -1.4032 -0.7015  0.9520 -2.8125
#> 
#> (6,.,.) = 
#>  -0.6160  1.8434 -1.7277 -1.7547  1.0645
#>   4.3311  0.0461  1.9562  0.0573  0.2527
#>  -5.7161  0.5723 -9.3385 -0.7597 -0.9824
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
