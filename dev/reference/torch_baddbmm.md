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
#> -2.1718  0.1806 -2.2391  1.9806  0.5982
#>   3.9641 -2.2727  2.5785  1.6440  1.9689
#>  -1.8812  3.3276 -0.7693 -3.9062  0.1914
#> 
#> (2,.,.) = 
#> -0.2796 -1.9076 -2.4533 -0.9762 -0.1775
#>  -0.1698  1.6642 -1.6263 -1.3013  3.3270
#>   2.2304 -0.3037  2.2348  0.3515  0.5743
#> 
#> (3,.,.) = 
#> -0.5138 -0.2785 -1.1520  0.0215 -0.2336
#>  -2.5581  0.6540  3.9680  1.5953 -1.1551
#>  -1.1571 -1.3448  1.3379  0.0455 -4.0379
#> 
#> (4,.,.) = 
#>  0.3220 -4.1653  0.3589 -1.4993  0.8570
#>  -0.4309 -1.4319 -0.1388 -5.1209 -0.9874
#>   0.9170 -1.8408  1.9213  4.1160  4.4669
#> 
#> (5,.,.) = 
#>  0.0411 -2.3003 -1.7137  1.6599 -0.8953
#>   2.5239 -0.8659  6.8203  1.7247  2.7899
#>  -4.8590 -1.9337  0.6573  1.0027  2.5609
#> 
#> (6,.,.) = 
#> -2.8647 -2.9735 -0.3530  0.5016  0.2747
#>   3.0110  2.8079 -1.3453 -1.2722 -0.4689
#>  -0.8511  0.2144  3.2881 -2.2362  1.7536
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
