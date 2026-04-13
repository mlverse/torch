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
#> -0.4309 -1.1955  0.1278  1.1805 -1.1589
#>  -0.1159  3.5347 -2.7812  1.0505 -0.2365
#>   0.2519  2.5645 -2.0098 -0.3014  1.2130
#> 
#> (2,.,.) = 
#>  0.4261  2.1847  0.4953  0.2701  2.4907
#>   4.7082 -2.9216 -1.7295 -3.2195 -2.8712
#>   6.4298  2.8515 -3.9759 -1.6025 -2.0636
#> 
#> (3,.,.) = 
#>  6.0284  0.9055  2.4453  3.7656  2.0584
#>  -1.0537 -0.1029 -0.1455 -2.5748  0.3786
#>  -1.8485 -1.3985  0.1153 -2.8361 -0.0267
#> 
#> (4,.,.) = 
#>  1.9557  3.7289  1.5692 -1.1769  4.4053
#>   2.2651  4.7236  2.5897  0.6591  4.2871
#>   4.9736  1.8945 -2.3761  0.3477  1.5018
#> 
#> (5,.,.) = 
#> -0.6466  3.7628 -1.5235  0.1901 -1.0438
#>  -2.5434  4.3050 -0.4789  3.0157  4.0251
#>   0.5342 -1.9644  1.1241 -3.8549 -2.4781
#> 
#> (6,.,.) = 
#>  2.4289 -1.5482  2.0971 -1.6863 -1.8867
#>   0.3243 -1.5953 -0.8669 -0.1800  3.0025
#>   4.0652 -0.9342  2.7010 -1.7342  1.6575
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
