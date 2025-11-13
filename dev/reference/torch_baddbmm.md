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
#>  -1.1310 -0.8982  0.4469 -0.6790  0.7342
#>   6.5273 -5.1402 -0.9146 -4.6279 -1.2451
#>  -0.6170  1.6565 -0.7325  0.9919 -1.0012
#> 
#> (2,.,.) = 
#>  -1.7691 -0.3294 -0.0902  0.5218 -1.1300
#>  -1.0556 -1.2054  0.7107  1.3577 -0.3570
#>   1.1103  2.7293  1.2616 -0.6136  1.7425
#> 
#> (3,.,.) = 
#>  -7.0233 -1.7407  1.0611  2.3366 -1.4185
#>  -4.9864 -1.1936 -0.6899  0.2320 -3.2459
#>   2.0814 -1.5702 -1.2726  1.1442  2.7276
#> 
#> (4,.,.) = 
#>  -4.4091 -3.5541 -0.5389 -3.3044 -1.6500
#>   0.4305 -1.6485  0.2796 -4.0693 -0.8893
#>  -6.8842  1.2654 -2.3738 -2.1080 -2.3907
#> 
#> (5,.,.) = 
#>  -1.7903  1.6051  0.2046 -1.1726  0.2249
#>   2.9395  1.3741 -1.9573 -0.9832 -0.1175
#>   2.9041 -2.3600 -1.5189  2.6982 -2.5412
#> 
#> (6,.,.) = 
#>   0.3179  2.6174  1.0329  0.7811  2.0786
#>  -1.4354 -0.3723 -1.4263 -0.2969  4.1726
#>  -1.1058 -2.8172 -2.0584 -2.9648  1.8230
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
