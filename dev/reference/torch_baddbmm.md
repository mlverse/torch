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
#> -0.6476 -3.7951  3.8791 -0.5708  3.7264
#>   0.3604  0.0941 -0.8949  0.0281 -2.9329
#>  -1.2048 -1.6243 -0.3759 -0.8554  0.3386
#> 
#> (2,.,.) = 
#>  2.5763  0.9208 -1.8443 -1.3055  1.0865
#>  -7.0371 -0.3243  2.0033  4.2180 -5.4648
#>  -1.8444  0.8213  1.0452  0.5947 -2.2176
#> 
#> (3,.,.) = 
#>  1.1787  1.0082 -0.5317 -2.2000 -3.5713
#>  -3.9603 -3.8660  3.4271  5.0162  3.8692
#>  -2.2727  0.2941 -1.3051 -0.6263  0.6832
#> 
#> (4,.,.) = 
#>  0.4322 -0.3769 -1.2526 -1.1264  0.1531
#>  -1.5310  1.8074  0.2924  1.4736  1.1504
#>   0.9128 -1.8871 -0.5453 -2.5500 -2.5189
#> 
#> (5,.,.) = 
#> -0.9157  3.1111  3.0414  3.9000  3.0016
#>   0.0819  1.5839 -1.7987 -4.4689 -0.6642
#>   2.0438 -2.7109 -3.6679 -1.9382 -0.3319
#> 
#> (6,.,.) = 
#> -4.4783  1.8762 -3.5289 -1.5731  0.4174
#>  -0.8554 -1.2589 -1.6062 -0.4598 -1.3856
#>  -1.2624 -0.4637  0.5214 -0.1744 -1.0394
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
