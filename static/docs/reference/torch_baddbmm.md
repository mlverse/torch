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
#> -0.4291 -0.0451 -0.0234 -2.8562  1.0754
#>  -1.1221  1.7270 -0.7340  2.1683 -3.4202
#>  -0.4131 -0.0361 -0.1769  2.5330  2.0333
#> 
#> (2,.,.) = 
#>  0.0300 -1.0880  0.6954 -1.6741  1.5388
#>  -3.3016  3.5389 -0.0468 -2.0052 -0.1474
#>  -0.9679  1.2148  4.1070  2.7039  4.5953
#> 
#> (3,.,.) = 
#> -1.3927 -0.8971 -1.8196 -0.0826  0.4828
#>   0.3134 -1.5259 -0.6889  0.5042  0.1259
#>  -2.9967  1.0487 -2.3676 -0.9442 -1.0846
#> 
#> (4,.,.) = 
#> -1.9370  1.2180  3.4276  4.0444  1.8076
#>  -0.4401  2.3587 -0.1682  0.7569  0.1294
#>  -3.4830  0.3061  0.0090  2.7728  0.1714
#> 
#> (5,.,.) = 
#>  1.0879 -0.4380 -0.7103  0.0379 -0.2275
#>   3.4571 -3.8095  0.4308  1.4208  2.8816
#>   1.0033 -0.7204  1.2447 -0.7856  2.3351
#> 
#> (6,.,.) = 
#>  1.5131  3.0786  1.7409 -0.0919  3.4090
#>   0.8848 -0.6703 -1.6173  1.0488  1.8397
#>   2.8039  0.1544 -0.2062 -0.5271 -0.8279
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
