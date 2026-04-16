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
#> -2.4952 -2.3559 -0.6471  4.1176  1.0384
#>   4.3584 -3.4251  0.2395 -1.1990 -1.4785
#>  -2.2896 -2.6543  2.5233 -0.6499  1.3342
#> 
#> (2,.,.) = 
#>  0.1664  0.0538 -1.0477 -2.1526  1.2281
#>  -0.8427  0.4522  2.1252  1.3949 -0.8556
#>  -0.1560  0.8704 -1.2339  0.9905 -2.5601
#> 
#> (3,.,.) = 
#>  0.3266 -2.0115 -0.0069  1.2593 -1.4160
#>  -2.0871 -0.5539 -1.0093  2.3318  1.5316
#>   2.1073 -3.7876 -1.5956  0.5973  0.1643
#> 
#> (4,.,.) = 
#> -1.0383  0.6977  0.5429  0.0791 -0.7102
#>   3.4201  0.9529 -1.1091  1.0504  2.7404
#>  -2.5505  1.0606  0.7275 -0.7502 -2.0399
#> 
#> (5,.,.) = 
#> -0.9250  0.9201 -0.7939  0.9126 -0.1979
#>  -0.9212  0.6357  0.0312 -1.0106  0.2375
#>  -1.4979  0.2481 -0.8604 -0.0593 -0.4821
#> 
#> (6,.,.) = 
#> -5.2378 -3.9800  4.7577 -4.0124 -3.5380
#>   0.0335  1.5629  0.7470  3.6989 -0.7577
#>   2.4085 -0.1888 -0.7844  0.6043 -3.6124
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
