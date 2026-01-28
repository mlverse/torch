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
#>  -9.5518 -4.0691 -3.6667  0.6246 -0.6392
#>   1.1694  1.5364 -0.1566 -0.6515  2.8504
#>   0.5368  0.1484 -1.3904 -6.1472 -1.3985
#> 
#> (2,.,.) = 
#>  -1.9813 -1.4599 -3.2125 -0.9748  0.5776
#>  -0.4768  3.1520  0.6815  0.4957 -1.1148
#>  -0.9638 -1.4225 -2.1129 -0.6455 -1.0506
#> 
#> (3,.,.) = 
#>  -0.4812 -0.1547  2.0832  0.7781 -0.2838
#>   0.6778  0.6048 -1.2845  0.5541  0.6321
#>  -0.0168  0.2906  0.3995  0.4285  0.7071
#> 
#> (4,.,.) = 
#>  -1.3617  0.1274  0.6091 -1.5397 -1.2429
#>   0.2526 -0.3865  0.9811  0.5484 -3.6396
#>   1.6558 -0.5602  1.8316 -1.8175  3.7845
#> 
#> (5,.,.) = 
#>  -0.2356 -1.5844  2.5831  0.7582 -6.1760
#>  -0.1661 -1.6825  1.9197 -0.6706  1.2872
#>   1.2127 -0.4842 -1.5261  0.2076  1.9239
#> 
#> (6,.,.) = 
#>  -1.0302 -0.4260 -0.9093  0.8128  1.1296
#>   0.2208  0.9890 -1.4037 -1.7985  1.0481
#>  -2.4438  1.2798 -0.2771 -1.4294 -2.3472
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
