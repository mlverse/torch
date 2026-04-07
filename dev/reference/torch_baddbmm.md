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
#> -0.7909 -0.4532  0.3274 -4.8094  1.9577
#>  -0.1293  2.5550  1.0296 -2.1533  0.8646
#>   0.8086  3.3215  0.8334 -1.5117 -0.9983
#> 
#> (2,.,.) = 
#> -2.7990 -0.0657 -0.1017 -5.5693 -3.5420
#>  -1.8445  1.8526  1.8658  3.2422  0.3205
#>   2.1490 -2.0622 -0.7042  1.7522 -0.9923
#> 
#> (3,.,.) = 
#>  1.8383 -0.6020 -0.1508  1.6146 -1.9528
#>   2.5460  1.0928  0.1424  2.7356  0.7375
#>   2.7397  1.6104 -0.2979  2.2509  0.9368
#> 
#> (4,.,.) = 
#>  5.4901 -0.5952  1.4927 -1.3618  0.6750
#>   3.9479 -1.3009  0.1163  1.2829  1.3308
#>   2.0490 -1.7852  0.9015  0.1572 -0.9141
#> 
#> (5,.,.) = 
#> -1.6333 -5.2425  1.0237 -5.4076  3.7803
#>   1.9832 -1.0163 -0.2323 -3.6893  1.4997
#>  -0.9728  1.3338 -0.5706  1.6219  1.3464
#> 
#> (6,.,.) = 
#> -2.8475  1.1295 -1.3896 -0.9200  1.1258
#>   1.4434  0.6607  0.8274  0.7011 -2.1768
#>   3.4038 -2.2700 -0.1539  1.3425 -0.7218
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
