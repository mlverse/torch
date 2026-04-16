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
#>  0.6276 -1.8067 -0.4876 -0.8137 -1.5886
#>  -1.0842  0.0740 -1.0442  0.2970  0.5012
#>   3.4478 -0.4693  4.2815  1.6517 -1.8900
#> 
#> (2,.,.) = 
#> -0.9702  0.2616 -1.0646 -2.5221  2.7291
#>  -1.0376  0.1258 -3.7873  0.1902 -2.3374
#>  -2.6274 -0.2705  0.0559 -3.5577  2.6854
#> 
#> (3,.,.) = 
#> -1.0841 -1.0985 -0.6977  6.3323  2.5743
#>   0.0686 -0.0411 -0.1849  1.7763  0.4816
#>  -1.9911  0.1102  1.3601  1.1291 -3.1461
#> 
#> (4,.,.) = 
#>  1.7049  0.3737  0.5514 -0.8404 -1.2214
#>   1.3051 -1.1843 -1.9803  5.0034 -1.6003
#>   1.0494 -3.3266  0.0736  0.7997 -0.2765
#> 
#> (5,.,.) = 
#> -0.5023  3.0417 -0.7218  1.5068  2.1743
#>   2.1169 -2.6502  1.8952 -4.0540  6.0868
#>  -1.6804 -1.1970  0.3398 -1.4676  3.3108
#> 
#> (6,.,.) = 
#>  1.5581 -2.8917  1.1817  0.7809 -0.0019
#>   1.9637  1.5005 -4.1293  1.9851  0.1774
#>   1.0085  1.9159 -1.1888 -3.9543  0.7730
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
