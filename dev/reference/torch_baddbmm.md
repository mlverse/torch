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
#>  0.8526  0.5103  0.5984 -0.0105 -2.7030
#>  -1.2162 -2.3652  2.5889 -0.5283  0.3074
#>   0.5219 -3.2222 -0.2738  3.7027 -0.6962
#> 
#> (2,.,.) = 
#> -1.0907 -0.2017 -0.3322 -1.8668 -0.2699
#>   0.6981  1.1032 -0.9821 -1.4283 -0.6264
#>   0.5574 -1.5689  2.5789  3.2091 -0.1536
#> 
#> (3,.,.) = 
#> -0.0159 -3.6784 -4.1324 -3.9695 -0.3293
#>  -1.5567 -0.5926  0.1786  0.0711 -2.0324
#>  -0.5056  0.1378 -1.3759 -0.6007 -2.4869
#> 
#> (4,.,.) = 
#> -2.6450  1.1855 -0.1757  1.3660  3.7188
#>   0.3290  1.5514  1.6577 -4.1864 -3.9291
#>   1.3417  0.2990  1.4370 -1.1707 -1.9879
#> 
#> (5,.,.) = 
#>  1.5177 -1.0298  0.0529  0.4709 -0.2998
#>   0.5435  3.0489 -1.0430  0.5750 -0.8477
#>  -5.5401  0.9812  1.0043  0.1708 -2.0373
#> 
#> (6,.,.) = 
#> -0.2854 -1.0745 -0.2428  4.3678  4.1381
#>  -0.0463 -1.4173 -0.4828  1.6943  0.9653
#>   0.5649  1.4529 -0.9782 -2.7231 -1.5626
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
