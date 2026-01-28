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
#>   0.3660  0.9406 -0.5290 -3.8055  0.7173
#>  -0.2214  4.1977 -1.2230  1.8297  5.3124
#>  -1.3451  1.8206  0.2709  2.7088  0.4750
#> 
#> (2,.,.) = 
#>  -0.0290 -0.1007 -0.0421  2.0432 -1.9338
#>   0.9927 -1.9930 -1.4006 -0.1522  1.0059
#>  -1.0699  1.5296 -1.8209 -0.4600 -0.9791
#> 
#> (3,.,.) = 
#>  -1.3254  2.4891 -2.3944  0.6094  1.8530
#>   0.7148  1.4226 -0.9139  1.1621  1.4224
#>   2.9984  1.3845  1.5289  0.4086 -1.9314
#> 
#> (4,.,.) = 
#>  -1.1915 -1.0230 -0.3352 -1.0281  0.2511
#>  -0.8452 -0.8148  1.2219 -0.9639  2.1850
#>   0.5357  2.1277 -1.1663  3.9925 -0.9917
#> 
#> (5,.,.) = 
#>  -1.5598  0.5811 -1.6317  0.8711 -0.5695
#>  -2.3202  1.9782 -1.4636  1.1992  1.1097
#>   1.3332 -1.2149 -2.1475 -0.8709 -2.5336
#> 
#> (6,.,.) = 
#>   0.2424 -1.6832 -0.5061  1.3274  3.9221
#>   2.8329 -0.4926  1.5735  4.6704  0.5390
#>  -0.2046 -0.9112 -0.0245 -0.5692  0.7039
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
