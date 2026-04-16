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
#> -0.4252  1.3920  2.0418 -3.1651  0.8460
#>  -0.4274 -0.5698 -2.8896  0.4666 -1.2379
#>   1.8304  3.1350 -3.8396 -0.8225 -5.7787
#> 
#> (2,.,.) = 
#>  1.6076  0.7303 -2.3051  3.0536 -1.4565
#>  -0.1719 -1.5550 -0.9594 -0.0559 -1.7068
#>   3.4031 -0.1897 -2.7554  2.9430 -6.1979
#> 
#> (3,.,.) = 
#> -1.3573 -4.3177  3.6084  2.1953 -2.5920
#>  -1.4672  4.0657 -2.2411  0.2631 -0.1591
#>   2.1312 -7.1331  4.8104  3.1133 -2.1920
#> 
#> (4,.,.) = 
#> -3.2801  3.6132 -4.0367  3.9036  1.0537
#>  -0.1562  0.4542 -0.1467 -0.9577  2.2717
#>  -5.2752  1.5622 -0.9983  1.1612  0.5839
#> 
#> (5,.,.) = 
#>  0.4550  0.4323 -0.1926 -0.5619 -0.2947
#>  -2.2860 -3.0910 -3.4343 -3.2816  0.7076
#>  -4.6837 -0.6937 -3.0879 -2.3886 -0.8088
#> 
#> (6,.,.) = 
#> -4.0838  0.0071 -2.3763 -2.0571  2.2195
#>  -2.9477 -1.6555  0.2177 -2.5953  1.3490
#>   6.9077  1.8405 -0.5388 -1.0446  2.8631
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
