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
#> -0.0003 -1.5293 -0.6615  0.1330  1.6696
#>  -1.0675 -0.8139 -3.2428 -3.3375  5.3943
#>  -1.7492  1.4683 -1.6458 -0.5770  2.3698
#> 
#> (2,.,.) = 
#>  1.1093 -0.8988  2.1657 -1.9252  0.6934
#>  -1.7976 -1.5636  2.6898 -1.1888  0.0169
#>   2.7467  3.8846 -0.0938 -0.1450  1.0241
#> 
#> (3,.,.) = 
#> -2.2846  5.3385  1.6947  0.4482 -0.0338
#>  -1.1058  4.9138 -0.6952  2.0475 -2.8545
#>  -2.8545  2.5282 -1.5191  0.3183  0.3792
#> 
#> (4,.,.) = 
#> -1.4389 -1.3918 -4.7027  4.8986 -1.7211
#>  -1.9250 -0.7562  0.8758 -0.0921 -0.0813
#>   0.0904  1.2976 -0.8397 -0.2767 -0.0817
#> 
#> (5,.,.) = 
#>  0.5458 -4.7521 -0.4464 -0.1002 -0.6320
#>   0.2920  5.1706  2.3975 -2.0144 -2.8589
#>  -1.1016 -0.7726 -1.3092 -2.0622  1.8651
#> 
#> (6,.,.) = 
#>  2.9251  2.0302  5.2612  5.8733  1.9697
#>   0.2196 -1.9474  1.7072 -0.9591  0.3724
#>  -2.0965 -2.3621  2.5166  0.5413  1.0554
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
