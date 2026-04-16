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
#> -1.9971 -1.3932  0.4735 -3.2759 -2.5057
#>   2.0834 -0.5677 -2.1742  1.0866 -0.3792
#>  -0.7896 -1.2506 -0.3691 -2.7722 -4.2164
#> 
#> (2,.,.) = 
#> -0.5327  1.1123 -0.6099 -1.5397  1.6425
#>  -0.6203 -0.1944  0.0330  0.1395 -1.0537
#>  -2.6460  1.5670  5.6024  3.9904 -2.7368
#> 
#> (3,.,.) = 
#>  1.0628  0.2345  1.1148 -0.5972  1.1436
#>   2.8805  0.7328 -1.3377  2.4223  2.0629
#>   0.1099  0.7854  0.0463  5.1603 -2.3701
#> 
#> (4,.,.) = 
#>  0.8172  2.5533 -5.0414  0.3656 -1.7685
#>   0.7033 -2.0608  1.5946  4.9978 -0.9917
#>  -0.9116  1.5235  4.1006  6.9255  0.7903
#> 
#> (5,.,.) = 
#> -1.9636 -0.4116 -0.1733 -1.6392  0.4529
#>   3.5095 -2.9631  1.7391  1.5606  0.6353
#>   0.1727 -5.8457 -3.1701 -0.6747  0.6148
#> 
#> (6,.,.) = 
#> -0.6257 -2.3467 -4.4191 -0.9700 -1.8938
#>   2.4434  4.1425  4.1273 -4.7785  3.1201
#>   1.2854 -0.6573 -1.7667  0.4761 -0.2955
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
