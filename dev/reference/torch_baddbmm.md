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
#> -0.6251 -0.0326 -1.6799 -1.4829  6.3439
#>   1.0960  1.3891  0.1237 -1.6059 -2.8171
#>   1.8588 -1.6905 -1.9175  0.3552  0.8568
#> 
#> (2,.,.) = 
#>  1.6929  5.8144 -0.0556  7.1297  1.4804
#>  -2.5897 -4.9841 -1.9862 -1.5694 -3.8465
#>   2.1370  6.0514 -0.7752  1.8599  0.7280
#> 
#> (3,.,.) = 
#>  1.2228 -1.5776 -3.2821  2.7808  0.2692
#>   0.0010 -1.5120  0.3088  1.7485 -0.6380
#>  -1.2812 -1.9201 -0.7549  1.6791 -1.8428
#> 
#> (4,.,.) = 
#>  1.0516  4.7821 -0.4124  2.3224 -1.6490
#>  -0.5216 -1.1532 -0.5626 -0.6674  0.6136
#>  -0.4741 -0.2318 -1.5752 -0.8650  1.6739
#> 
#> (5,.,.) = 
#> -1.2268 -2.7779 -0.4274  0.4321  2.4380
#>   0.8107 -0.4250  3.5666 -0.0089 -1.6407
#>  -1.7844  2.1541 -2.6205 -0.1430  3.1304
#> 
#> (6,.,.) = 
#>  1.4281  1.6050 -0.6469 -1.6931  1.9591
#>   0.6032 -2.2135 -2.4094  4.1821 -1.3936
#>   2.5971  0.0126 -0.9266  0.4967  1.7458
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
