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
#>   5.3737  0.5537 -2.9056 -0.9553 -1.1379
#>  -0.3004 -2.3345 -1.0091  0.8250 -1.2171
#>   0.5070 -0.9175  1.9591  0.8996  1.2991
#> 
#> (2,.,.) = 
#>   1.2332  2.7090  1.0723 -3.1322  2.0471
#>   0.0357  1.6460  2.2362 -4.1669  3.6877
#>  -2.8228 -3.7610 -2.9767  0.1559 -3.3788
#> 
#> (3,.,.) = 
#>  -0.4197  0.1740  0.0820  0.5327  1.7687
#>  -2.1068 -0.9199  0.7976  0.1537  2.7188
#>   0.0159 -1.7314 -2.3839 -1.0057 -1.0043
#> 
#> (4,.,.) = 
#>   5.7855 -2.3927 -0.1915 -2.8672  1.6920
#>  -1.3269 -2.5068 -2.3956  1.2278  1.0407
#>   4.6396  4.1262  2.3907 -2.8922  1.2153
#> 
#> (5,.,.) = 
#>  -0.1599  0.8616 -0.4773  3.8296  0.8295
#>  -1.3583  0.1612 -1.6117  1.3179 -1.7062
#>   0.0088 -1.8956 -1.7811  2.0982 -0.2113
#> 
#> (6,.,.) = 
#>   1.6685  0.6296 -1.2355  5.1991 -1.9642
#>   1.4851  0.9077  1.1203 -0.3804 -0.3993
#>  -1.9132 -1.7437  3.4388  0.6463 -3.3174
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
