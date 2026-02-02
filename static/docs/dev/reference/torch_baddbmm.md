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
#>  -2.3558 -0.4845 -1.0841  1.9239 -1.0724
#>  -1.2129 -3.2748 -4.7269 -0.1253  2.8437
#>  -0.0720 -0.9056  0.1754  2.3494  0.1805
#> 
#> (2,.,.) = 
#>  -1.3201  1.8426 -1.0548  0.3592  0.1498
#>  -1.3013  3.1952  1.8209  0.1599 -0.8697
#>  -3.7048 -3.6472 -0.5401 -2.6848  1.3056
#> 
#> (3,.,.) = 
#>   1.1648 -0.5009 -1.3661 -0.4308  0.6346
#>  -1.6445  1.9060 -2.0505  0.2200  0.6839
#>   1.1505  2.1274 -1.1377 -1.0147 -2.7278
#> 
#> (4,.,.) = 
#>   0.5951 -0.8791 -0.3892 -0.2805  2.4389
#>  -0.2862 -1.4416  2.8000  1.4252  3.2233
#>  -1.7025  0.4566  2.0807  2.6453 -2.4276
#> 
#> (5,.,.) = 
#>   2.4246  2.2070 -1.3499 -0.8624 -1.2088
#>   2.2835  4.7170 -0.5084 -4.6877  2.1311
#>   1.3130 -1.9010 -2.7068 -0.9835  1.3700
#> 
#> (6,.,.) = 
#>   2.1688  1.2451 -1.2952  1.7441  2.2253
#>   2.1670 -1.2272  0.1624  2.5229  0.1695
#>  -0.3463 -0.5984  0.4572 -0.1897  0.7933
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
