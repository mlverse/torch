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
#> -2.2206  5.0824 -2.5514 -1.7373  1.2373
#>   2.8832 -3.7850 -1.7479  1.1672  1.8237
#>  -0.2406 -2.6194  0.0473  0.6357  3.7498
#> 
#> (2,.,.) = 
#> -2.0884  2.5341  3.0376 -3.2435 -0.5832
#>  -1.9462 -0.8222  0.3511  0.2818  3.0797
#>   5.5836 -4.6112 -2.9017 -0.2662 -2.9683
#> 
#> (3,.,.) = 
#> -0.3611  0.2103 -1.7008 -1.1564  1.7874
#>   1.5787 -0.8179  1.9302 -0.0577 -3.4364
#>   2.2389  0.2697  1.4734 -2.3865 -2.0521
#> 
#> (4,.,.) = 
#> -2.9172 -2.1355 -3.2449 -0.5507  4.6256
#>  -0.7838  0.8850 -0.4683  0.5368 -1.8584
#>   2.0561  0.0992 -0.7447 -0.1491 -0.4119
#> 
#> (5,.,.) = 
#>  1.7497  2.0847  0.8347 -2.7218 -1.2374
#>   1.6616 -5.6754  3.0132  3.4505 -4.2577
#>  -2.5850  5.5577 -4.5387 -2.5370  5.2635
#> 
#> (6,.,.) = 
#>  0.0613  4.1707  0.9041 -3.8781 -2.7040
#>  -1.5305 -1.6138 -1.2135  2.2483 -2.4444
#>  -1.2212  0.3878  0.8321  0.1402  1.0549
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
