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
#>  -1.9046 -0.5389  0.2426 -0.5616  4.9992
#>  -2.9425 -1.3861  1.3829  1.0929  1.9556
#>  -1.8080 -1.2925 -1.2781 -0.6309 -0.0925
#> 
#> (2,.,.) = 
#>  -1.6669 -1.6280 -2.6720 -6.3823  3.3698
#>  -3.0307  1.4500  2.1941 -2.8958  0.4748
#>  -0.4435 -0.2324 -0.0891 -1.3765  0.7520
#> 
#> (3,.,.) = 
#>   4.8434  1.2792  0.7686 -1.4778 -2.6972
#>   2.3688 -0.8735  3.1199  0.0356 -1.9977
#>  -1.9794 -0.2445 -8.1998  2.2800  2.6660
#> 
#> (4,.,.) = 
#>   0.1982 -4.6086  1.2850  1.7967 -0.5887
#>   3.4304  2.7238 -1.4750 -1.7413  1.9256
#>   0.0704 -3.6472  0.5212  1.0868 -1.9360
#> 
#> (5,.,.) = 
#>   1.0710  1.5211  1.8333  0.4814 -0.5528
#>   0.1914 -0.7333  0.2067 -0.7256 -1.5215
#>  -1.1854 -1.1253  0.3765 -1.4202 -0.2329
#> 
#> (6,.,.) = 
#>  -2.1864  0.0746 -0.4959  1.2990 -1.5800
#>   0.2413  4.9997  6.6288  3.0751 -1.9148
#>   1.4678 -2.3510  2.4991 -1.5524 -2.9018
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
