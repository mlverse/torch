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
#>   0.2339 -0.2908 -0.2834 -0.5535  1.2120
#>  -0.6190 -0.4845  0.5931  1.7559  0.1178
#>   0.2469  3.1206 -0.0199 -0.9272 -0.8002
#> 
#> (2,.,.) = 
#>  -1.7035  0.5755  2.1283 -0.6168  1.9119
#>   0.4176  4.0062 -0.3949 -1.5953  2.4803
#>  -0.4327  8.0358 -2.2617 -1.9540  2.1837
#> 
#> (3,.,.) = 
#>  -1.5529  0.0606  0.5024  1.0160 -1.0737
#>  -0.3749  0.5000 -1.0608 -0.4966 -0.0840
#>   1.4185 -1.2671 -0.1367  1.5834  2.6444
#> 
#> (4,.,.) = 
#>   1.0997  1.8018  1.8928  2.5728  0.5473
#>  -2.8994  2.0650 -3.7355 -0.6053  2.5942
#>   3.2950 -1.1888  0.1430  1.7150 -3.1134
#> 
#> (5,.,.) = 
#>   3.9627  1.9117 -2.6322  2.8821  3.6160
#>  -2.5727 -1.2211 -0.8723 -0.4506  1.2266
#>   1.4611  3.0395  0.6718  0.9234 -5.7276
#> 
#> (6,.,.) = 
#>  -1.4463  1.8235 -0.2778 -0.5290  0.9482
#>   0.2469 -0.7224  0.3534 -0.6067 -0.5692
#>   3.0639 -7.8497  2.9629 -1.1496 -1.1398
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
