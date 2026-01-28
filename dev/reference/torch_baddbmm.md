# Baddbmm

Baddbmm

## Usage

``` r
torch_baddbmm(self, batch1, batch2, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to be added

- batch1:

  (Tensor) the first batch of matrices to be multiplied

- batch2:

  (Tensor) the second batch of matrices to be multiplied

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
#>  -1.6861  4.1822  1.2571  2.4451  0.3718
#>  -2.3228  1.8323  1.6658  1.2122 -1.0446
#>  -3.8161  4.7041  2.4638  3.8760  0.5253
#> 
#> (2,.,.) = 
#>   0.9680  0.4005  1.2793  0.6141  1.2078
#>   1.4009  2.9900 -2.4577  1.7780 -0.7544
#>  -0.4990 -1.0539  1.5074  0.0579  0.9507
#> 
#> (3,.,.) = 
#>   4.0414  4.2045  4.0478 -0.8902 -1.4405
#>   0.9372  0.7718  2.2848 -2.3033 -5.1426
#>   0.8461 -1.9774  0.1930 -0.1229 -1.8561
#> 
#> (4,.,.) = 
#>   5.0264  4.3420 -1.1117  2.4826 -1.9867
#>   0.8611  1.0903 -0.1251  0.8952  1.3667
#>   0.1179  1.0832 -5.0138 -1.4450  1.1996
#> 
#> (5,.,.) = 
#>   2.0863  0.0622  3.6935  2.3777 -2.5942
#>  -0.3627 -1.2425 -1.3129 -0.8641  1.1807
#>  -9.2277  6.2822 -2.7658  2.5313 -1.3903
#> 
#> (6,.,.) = 
#>  -1.1778  0.7300  3.3005  0.9922 -1.3961
#>  -1.6927 -1.8684  2.0424  2.5226  2.6298
#>  -0.8935  2.9528  2.2081  0.6403 -0.9384
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
