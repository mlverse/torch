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
#> -0.5170 -1.1721  1.5030 -2.1872 -4.0760
#>   1.9696 -0.4143  1.2173  0.3465 -5.0101
#>  -1.3426  2.3968 -1.7825 -1.3937  1.3174
#> 
#> (2,.,.) = 
#> -1.5998 -0.0009  2.8534 -1.1403 -1.7635
#>  -3.2290 -2.0241  0.6925  0.5921  1.4625
#>  -0.9205 -1.9142  1.5930  1.5895 -0.7850
#> 
#> (3,.,.) = 
#> -0.3638  0.5383  0.5160  0.0987 -0.1547
#>   0.7890 -2.6544 -0.1245 -5.6616  2.7483
#>  -3.1689 -0.4773  2.5752  3.1090  0.6949
#> 
#> (4,.,.) = 
#> -1.7233  4.9360 -0.6662  0.0763 -0.4124
#>   1.1026 -2.1880 -0.8633 -0.4063  1.3607
#>  -2.1212  0.7316 -1.6488  0.5836 -0.3036
#> 
#> (5,.,.) = 
#> -1.3462  0.3122 -0.9398 -1.1364  1.8670
#>  -1.3016 -1.2067  1.9018 -2.3399  2.5142
#>   2.0595 -0.0314  2.4626  1.4520 -1.3583
#> 
#> (6,.,.) = 
#> -2.5612 -2.2246 -2.8311  0.4507 -2.2108
#>   0.6504  1.2500 -1.8988  1.0997  0.3499
#>   2.4983 -1.0928 -1.0447  0.8651 -1.5316
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
