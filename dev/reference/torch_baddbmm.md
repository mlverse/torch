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
#> -3.2308  2.6763  2.8525  1.5920  1.4289
#>   2.4342 -1.6653 -0.8782 -3.1559  2.6015
#>  -0.1371  3.1287 -1.8442  0.0167  3.1310
#> 
#> (2,.,.) = 
#>  3.1079 -0.1736 -2.1575 -3.7777  1.9439
#>   2.5861 -1.2443  1.9425 -0.5892 -2.9975
#>   0.4190 -2.4118  4.6866  1.2965 -1.5805
#> 
#> (3,.,.) = 
#>  2.4425  0.7161  3.3493  3.4080  1.2547
#>   0.0817 -2.3273 -2.2413  1.7982  0.2803
#>  -1.7884 -0.1395 -0.6101 -3.0361 -2.9274
#> 
#> (4,.,.) = 
#> -1.8943 -3.2978  4.9495 -1.6611 -3.8480
#>  -0.8178  1.8020 -0.8567 -2.5862 -1.3282
#>   0.2208 -2.3255  0.4112  0.7489 -0.1746
#> 
#> (5,.,.) = 
#>  0.3739  0.3861 -0.5763 -1.1741  0.3128
#>  -0.3847  0.3765  1.6248 -2.7091  0.9636
#>   2.9137 -1.3300  2.6258 -1.0725  3.4173
#> 
#> (6,.,.) = 
#> -2.2249  2.3128 -1.7141 -2.3345  0.6070
#>   1.8356  3.0843  0.0973  1.5866  0.5047
#>  -3.1701 -0.2658 -4.0952 -0.8107  0.6467
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
