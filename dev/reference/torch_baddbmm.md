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
#>  1.2070  0.3266  1.0614  3.5800 -0.8813
#>  -0.8327  4.0110  1.8812 -0.3272  1.4384
#>  -1.2608 -1.0205 -0.0341 -1.5508  2.2727
#> 
#> (2,.,.) = 
#>  1.1275 -2.4317  0.8501 -1.4448  1.3956
#>   1.6027  3.5779  3.0359  1.0942 -4.7947
#>  -0.8904 -0.4789  0.2448 -0.6439  1.7664
#> 
#> (3,.,.) = 
#> -5.7114 -3.1119  2.2772 -0.7228  1.1123
#>   0.3871  2.7519  1.0828 -1.3722 -1.3211
#>  -4.3642 -6.5420  3.0072  0.1265 -2.7939
#> 
#> (4,.,.) = 
#> -0.4183 -1.7263  2.0861 -1.5105 -1.1831
#>   1.2003  1.4011 -0.6030  0.5829 -0.2103
#>   0.5717 -2.3398 -1.3533 -0.7408 -0.8817
#> 
#> (5,.,.) = 
#> -1.1250 -1.1005 -1.1267 -0.7838  2.3344
#>   0.9761  1.8475  0.2719  0.3317  0.4665
#>   1.6351  6.3853  0.5372  3.2203  0.3201
#> 
#> (6,.,.) = 
#>  2.0791 -2.3847 -0.1529  1.5068 -2.1359
#>  -0.7475  2.8151  2.2065 -1.9218  1.6107
#>   2.2830  3.6288  2.0956 -0.1119  3.3205
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
