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
#>  -1.1864 -2.6154 -3.7572  1.1540  0.7709
#>  -1.4664  1.5488 -1.0400  0.9116  0.4954
#>   4.5382  1.0784  1.1850  0.3220  1.7774
#> 
#> (2,.,.) = 
#>  -2.4152  0.6077  0.5708 -1.8694 -1.0812
#>  -0.0308 -0.2850  0.1902 -2.0314  1.0112
#>  -5.4978  1.0617  0.8981  2.4483 -2.7882
#> 
#> (3,.,.) = 
#>   6.5171  0.7091 -2.5257  5.1976 -2.7261
#>  -2.7954 -0.8163 -1.3252  0.0183  1.9375
#>   2.9235  0.6662 -2.8976  7.2738 -3.1294
#> 
#> (4,.,.) = 
#>   1.1889 -0.0981 -0.2057  1.2104  4.4915
#>   1.4879  0.9983 -1.3937 -0.2874 -0.5276
#>  -0.7634  1.3706  1.5283  1.1306 -3.4740
#> 
#> (5,.,.) = 
#>  -2.6681 -1.3095 -0.1390 -4.2347 -1.3495
#>  -1.0912 -0.2469 -1.6568 -3.6284  0.5982
#>  -3.5306  1.4584  1.7493  1.7600  0.2378
#> 
#> (6,.,.) = 
#>   0.8047  2.3173 -3.3881  0.4941  0.3483
#>   0.7947 -3.3711  0.3682 -1.7363 -4.0713
#>   0.5809  0.7524 -0.5862 -0.2249 -0.3865
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
