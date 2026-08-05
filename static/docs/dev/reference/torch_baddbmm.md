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
#> -2.2578 -1.6984 -0.8088  0.5057 -2.0870
#>  -1.0377 -0.9329  0.2562  3.8437 -0.5845
#>   1.6623 -0.5774  2.2921  1.7251 -0.4398
#> 
#> (2,.,.) = 
#>  1.2023 -1.1849 -2.0297  5.8136 -0.2254
#>  -4.9121 -0.6021 -2.9232  2.4739 -0.2550
#>   5.1297 -0.4956  0.0501 -1.5461 -0.7329
#> 
#> (3,.,.) = 
#> -0.8222 -2.4759  0.5677 -1.6353  1.8933
#>  -2.8677 -2.5328 -1.7032 -2.1597  4.6964
#>   0.8590  1.8256  0.1439  2.0254  1.2205
#> 
#> (4,.,.) = 
#>  1.3831  0.4091 -1.3633  1.6682 -1.6902
#>   0.1674  5.2069 -0.2237 -1.2783 -0.7384
#>   1.2114  1.6355 -0.6092 -1.0562  3.1015
#> 
#> (5,.,.) = 
#>  1.2762 -0.2929  2.3092 -1.1619  1.1300
#>   0.2454  3.4614 -0.3920  0.4077 -1.1543
#>  -0.2401 -0.4306 -1.4069 -2.6095  0.1962
#> 
#> (6,.,.) = 
#> -0.1001  2.5016  1.5585  2.0022 -1.0864
#>   1.6383  0.1652 -0.4612 -0.4561 -1.4804
#>  -2.7941 -0.0336  1.7726 -1.5696  0.6527
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
