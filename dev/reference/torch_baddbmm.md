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
#>  1.6130  0.7190 -0.8850 -1.1639 -0.3615
#>  -0.5746  1.9189  1.0040  1.1281  1.7271
#>   1.6991  1.0520  0.2078 -0.4820 -1.0798
#> 
#> (2,.,.) = 
#> -0.9845 -0.6142  0.0805  0.1249  1.1018
#>   2.0778 -0.1625 -2.0966  2.1338  2.3970
#>  -0.4452  0.8938 -1.8135 -0.6064  1.0360
#> 
#> (3,.,.) = 
#> -0.5882 -1.4974  0.7798  0.5679 -1.6236
#>  -1.0472  0.0826 -0.1841  2.1697 -1.8044
#>   2.9845 -1.7692 -0.0078 -0.9541  2.5648
#> 
#> (4,.,.) = 
#> -3.9687 -1.6522  4.0433  0.4605 -0.6287
#>  -1.1524  1.1063  2.3822  2.2978  1.3771
#>  -0.3266  2.0091 -2.7980 -0.3116  2.0998
#> 
#> (5,.,.) = 
#> -1.5346 -0.0207  0.4047 -1.6923  4.3067
#>   1.3178 -0.9816  0.9305 -0.6574  0.8419
#>  -0.4767  3.9592 -2.7905 -0.8243 -3.2262
#> 
#> (6,.,.) = 
#> -2.2064  0.0313  1.4654 -0.6666  1.0218
#>   1.3001 -4.5609  1.8104  3.0594  1.2340
#>  -0.5642  1.4478 -0.8081 -3.1733 -2.1822
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
