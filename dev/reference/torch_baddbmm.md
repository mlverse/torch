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
#>  3.2581 -0.4020  0.9626  0.0452 -0.8878
#>  -0.9557  2.5568 -0.0929 -2.5757  1.6375
#>   1.5233 -0.6939  0.7489 -1.2535  4.0860
#> 
#> (2,.,.) = 
#>  0.3870 -2.3167  0.9202 -3.8205  1.8509
#>  -1.6983 -5.4971  1.5209 -6.9344 -1.6764
#>   1.0183  0.0653  0.7303  0.1384 -1.4596
#> 
#> (3,.,.) = 
#>  2.3489 -2.0631 -6.3462  7.0188 -0.4534
#>  -2.7022 -1.3396  1.7866 -1.0451 -0.6233
#>   0.2566  0.5402 -0.3836 -2.5173 -4.4669
#> 
#> (4,.,.) = 
#> -2.1246 -4.9566  0.3146  3.5670  1.5042
#>   4.0169 -2.3746 -0.7724  2.0821  6.3185
#>  -2.9352 -2.3984  2.7657  1.9716 -0.0730
#> 
#> (5,.,.) = 
#>  1.2740 -2.8331 -1.7451  3.2452  1.9679
#>  -2.6966 -1.1787  1.3062  4.8024 -1.3568
#>  -5.4892 -0.8374 -0.5929  2.5026 -6.1931
#> 
#> (6,.,.) = 
#>   3.4146 -10.1638   0.8835   1.4439  -3.0096
#>   -4.6958   2.7179  -3.5343   2.6321  -2.2776
#>    6.3342  -8.4584   4.6716  -2.8542  -1.3102
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
