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
#> -3.9465 -3.2697 -2.2550  0.5274  4.0525
#>   3.8098  1.2706 -0.4927  1.2524 -4.8060
#>  -4.5387 -0.0018 -4.7190 -2.3855  3.0559
#> 
#> (2,.,.) = 
#> -0.5851 -6.0703 -6.0763 -0.2281 -1.2489
#>   0.1233 -3.7684 -2.5086  0.9708 -2.1506
#>   1.8653  1.0367  3.6076  1.5662  1.8633
#> 
#> (3,.,.) = 
#> -2.8053  0.1504 -0.4003 -1.2701 -0.8897
#>   2.1462  0.8154  2.3855  4.6182  0.9026
#>  -2.5114  1.9966  1.1331 -4.6787 -0.3059
#> 
#> (4,.,.) = 
#>  0.9088  0.1974  2.1903 -1.6514 -0.8917
#>   2.3557  0.4276 -0.5913  1.3501  2.1275
#>   0.9919  2.3944 -1.9722 -4.1107 -4.4486
#> 
#> (5,.,.) = 
#>  0.9444 -0.5980  0.4569 -1.0980  1.0486
#>   0.1615 -3.0211  5.4210  0.6016 -2.9856
#>  -1.1389  2.7259  1.5420  2.5865  0.6683
#> 
#> (6,.,.) = 
#>  0.8064  0.2508  0.7668  0.5980 -0.7979
#>   0.6830 -4.6111 -1.8025 -2.9807  8.2240
#>  -5.2020  0.3575 -1.0230 -0.1651  0.8551
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
