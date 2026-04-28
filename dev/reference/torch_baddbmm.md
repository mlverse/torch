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
#> -1.5666  0.7208 -1.7318  0.7676 -0.1731
#>  -0.1343  2.6503  0.8653 -1.7114  0.5544
#>  -0.0313  3.1082  1.3434  0.9205 -1.1990
#> 
#> (2,.,.) = 
#>  0.4247 -0.1377 -6.6516  1.7574 -3.3893
#>  -0.6162 -0.8086  1.2755 -1.6773 -1.9411
#>   1.6297  2.0449 -2.8886  0.2773 -2.1403
#> 
#> (3,.,.) = 
#> -3.3442  0.7348 -0.4730 -0.3151  0.5421
#>   2.6819 -1.0154  1.8773 -1.2890  0.3475
#>  -0.9228  3.8824 -0.0307 -0.6487  0.2114
#> 
#> (4,.,.) = 
#> -2.3500  2.3106 -1.8508 -5.2346 -4.3977
#>  -2.5683  1.4441  0.7057 -1.5351 -6.6261
#>   2.0178  1.1649 -7.7295 -0.6973  0.5404
#> 
#> (5,.,.) = 
#> -1.3558  1.2802 -1.4988 -3.5438  5.7329
#>  -0.3534  1.1709  1.1282 -1.5993  3.6493
#>  -0.0040  0.2423  1.6824  0.9509 -1.1124
#> 
#> (6,.,.) = 
#>  2.0872  2.9759  0.8474  0.4906 -2.7390
#>  -1.3684  3.6905 -1.4893  2.0079  4.7996
#>  -1.2116 -1.4476  1.2484 -1.2843 -0.2150
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
