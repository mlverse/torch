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
#>  0.5176  2.1729  0.9159  1.8811  0.1900
#>  -5.1919 -1.6450  1.2753  4.0854  1.0859
#>  -0.7494 -2.2152 -0.2256 -0.7431 -2.1843
#> 
#> (2,.,.) = 
#>  2.3956  4.1407  0.5506  2.8578  4.5827
#>   0.5313 -0.4144 -1.3010  1.4953  4.2784
#>  -0.1953  3.0599  0.7033  0.7998 -0.9855
#> 
#> (3,.,.) = 
#>  0.3118  2.4716  1.8379 -0.6547 -0.5033
#>   0.9610  0.8605  1.1430  1.5278  0.7658
#>  -3.8363 -0.1611 -1.4087  3.2200 -1.1539
#> 
#> (4,.,.) = 
#> -0.1535 -2.6262  1.9698 -3.7730  0.0706
#>   0.7012 -5.8706  0.7319 -6.0131  1.4529
#>  -1.7592 -1.9047  1.6649  1.0226 -1.6940
#> 
#> (5,.,.) = 
#> -3.5145  0.1057  1.2012 -2.9224  0.3272
#>  -3.5095 -0.5936 -1.4969 -0.2884  2.5086
#>  -0.7244  0.4009 -0.3157  0.3529 -0.9527
#> 
#> (6,.,.) = 
#> -1.8958 -1.4449  0.5164  0.0101 -0.7792
#>  -0.5234  1.4038 -0.4293  0.8460  0.9874
#>  -0.1102  3.0275 -1.6948 -1.5688 -0.4800
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
