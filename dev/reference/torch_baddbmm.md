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
#> -0.1515  0.1234 -4.4307  2.5839 -0.8503
#>  -1.0340 -2.3372  1.7530 -2.1783  3.3580
#>  -1.6493  0.2127 -2.4171 -0.5905  3.1331
#> 
#> (2,.,.) = 
#> -2.7662  4.3460 -1.2052  1.4722  0.2992
#>   0.1787  1.1835 -1.5674 -2.4342 -2.8750
#>   2.3711 -3.2745 -0.7261 -0.9337 -2.0096
#> 
#> (3,.,.) = 
#> -3.6175 -0.1904  0.4639  0.7723 -1.9730
#>   0.2900 -2.0581 -0.9181 -2.3757  1.0457
#>  -0.7287  0.0682 -2.9693 -1.4878 -0.1943
#> 
#> (4,.,.) = 
#> -1.4575 -0.3343 -2.5752  1.0937 -4.9167
#>   0.2710 -0.1761 -0.1806 -1.5927  2.3101
#>  -2.0397 -0.0100  0.4872 -1.2698  0.1581
#> 
#> (5,.,.) = 
#>  0.0640 -1.0402  0.7263 -2.8096  1.1956
#>  -0.9638 -0.2627  3.8956  3.0360 -3.5386
#>  -0.2873  0.2908  4.1344  2.8876  0.1345
#> 
#> (6,.,.) = 
#>  1.7020 -1.6193 -4.8556 -1.9663 -0.9562
#>  -0.0036 -0.9553  1.8583  1.5256  0.0213
#>   1.0277 -1.2834 -9.3794 -4.4462  3.9409
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
