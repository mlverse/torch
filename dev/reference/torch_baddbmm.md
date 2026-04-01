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
#>   1.4356  4.6424 -3.5967  0.1782 -1.7491
#>   1.8388  4.2609 -1.9826  0.4863  2.5192
#>   0.3608 -0.1256 -0.1420  1.8610 -2.7605
#> 
#> (2,.,.) = 
#>   1.2303  0.4179 -2.0715  0.6806 -4.9097
#>  -1.3236  1.1496  1.8241 -2.2844  2.2942
#>   0.1635 -0.9005 -0.2521 -2.0195  1.5155
#> 
#> (3,.,.) = 
#>  -0.4513 -2.7722 -2.5020  0.9386 -1.3717
#>   0.0370  0.1569  2.8485  0.1773  2.7052
#>   0.2579  1.7273  4.2580 -0.8949  0.3930
#> 
#> (4,.,.) = 
#>  -0.1674 -1.2259 -2.6162  3.4203  3.6836
#>   2.9073  1.7808  1.2361 -4.0118  2.2241
#>  -0.4817  3.9172  3.0796 -3.3791 -1.7525
#> 
#> (5,.,.) = 
#>  -1.0681  1.7737  1.4131 -3.4921  0.8855
#>   0.7944  3.5058  4.5536  0.6829 -1.3137
#>   0.5900  2.3089  0.8629  1.9593 -1.0674
#> 
#> (6,.,.) = 
#>   1.6911 -2.6761  0.0614 -1.4084  2.2118
#>  -1.2444 -1.0780 -1.4663  0.6261 -1.7228
#>  -2.8314 -1.0966  1.8156 -1.0726  1.2636
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
