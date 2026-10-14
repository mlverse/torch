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
#> -3.7309 -1.2067 -1.6885 -0.0285  4.0822
#>  -1.3476  0.6941 -1.7840 -1.5789  0.2994
#>   1.2016 -0.4712  3.8883  0.5456  0.2072
#> 
#> (2,.,.) = 
#>  0.3413 -0.7724  0.3475 -0.5200 -0.8723
#>  -2.0818  2.4121  0.5072  1.2665 -2.1911
#>  -0.6957  0.2879 -1.9326  3.7198 -3.5172
#> 
#> (3,.,.) = 
#>  0.3218  1.1071 -2.5312 -0.1967  2.1118
#>  -4.2764 -5.9710 -0.5371 -2.3911 -1.6842
#>   0.6814  3.0519  0.3022 -0.9406  0.6411
#> 
#> (4,.,.) = 
#>  0.0674  0.1919 -2.2624  1.0077 -0.8555
#>   1.2109  1.1787  2.8484 -1.5142 -0.3002
#>   0.1155  4.4872  1.5464  0.2625  3.5942
#> 
#> (5,.,.) = 
#>  1.0299  0.1524 -0.6346  0.6229 -1.4333
#>   0.4920  1.3070 -2.6602  1.0514 -0.3682
#>  -0.1297  1.6932 -0.6874  1.8863  1.2150
#> 
#> (6,.,.) = 
#>  0.0966  0.2520 -3.3572  3.2678 -2.7944
#>  -0.3827 -0.7018 -0.4590  3.6940 -1.4943
#>   2.6756 -0.1030 -0.4441 -0.7800 -0.6658
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
