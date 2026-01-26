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
#>   0.6741 -0.7056  1.0796 -1.0815 -0.4185
#>  -0.9816  1.0164 -4.3286  1.0483 -1.7986
#>   0.2937 -1.5020  1.6206  2.6234  0.5048
#> 
#> (2,.,.) = 
#>   1.0389  2.2528  2.2692  2.6016 -2.0916
#>   1.6011  0.8905 -0.1666 -2.9913  0.0603
#>  -4.3989 -1.1841 -0.4643  0.7554 -3.3074
#> 
#> (3,.,.) = 
#>  -1.3624e+00 -2.7885e+00  1.6591e+00 -8.5860e-05 -8.7985e-01
#>  -2.3539e+00 -3.2569e+00  2.5720e+00 -1.6106e+00 -1.7405e+00
#>   8.8027e-01  2.9155e+00 -2.9180e+00 -1.9147e+00 -4.7424e-01
#> 
#> (4,.,.) = 
#>  -0.8852 -4.0711 -2.5883  2.4904 -1.5194
#>   0.8561  0.6158 -0.4361 -0.4060  0.0030
#>   0.7843 -0.4127  0.2309  1.6889  0.9828
#> 
#> (5,.,.) = 
#>  -1.7756  1.5314  0.9703  2.1164  3.3532
#>   0.0780 -1.6622 -1.6370 -2.9592  0.5342
#>  -0.1977  0.6391 -2.8616 -1.9416 -10.5702
#> 
#> (6,.,.) = 
#>   0.8631 -3.9591  1.5709 -0.7855 -0.2292
#>  -0.9950 -1.4453 -0.8576 -1.3411  0.0682
#>   0.4768 -3.4964 -5.1632 -2.4823 -2.7597
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
