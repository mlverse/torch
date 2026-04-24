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
#> -0.5523 -0.5237  1.3044 -0.2358 -3.1425
#>   1.4652  2.4789 -1.9341  0.2347 -0.0854
#>  -1.3236  0.2594 -0.8909  1.0445  1.9574
#> 
#> (2,.,.) = 
#> -2.1870  2.2839 -0.6492  1.4999 -1.5379
#>  -0.4685  0.7041 -0.0527 -0.5842 -2.8899
#>   1.4570 -1.9492 -3.4844  0.6198  2.1642
#> 
#> (3,.,.) = 
#> -1.6653  0.9954  0.1349 -1.0692  0.9651
#>  -0.0707  0.0787  0.9820  1.6115  0.5159
#>  -0.8392  0.4082 -0.8110 -3.9977  0.0100
#> 
#> (4,.,.) = 
#>  1.2357 -0.4601  1.7843  0.1777  0.0212
#>  -0.6235  1.0719  1.1302 -2.7335  2.2412
#>  -1.7370  1.7097 -2.5676  0.7680 -0.9347
#> 
#> (5,.,.) = 
#> -3.6664 -0.2516  1.7518  1.3586 -4.1132
#>   2.0186 -3.7753  1.6874 -0.3932  4.8142
#>   0.1266  0.1232  2.0343  0.9761 -1.8882
#> 
#> (6,.,.) = 
#> -0.0791 -0.5860  1.5940 -1.6987 -1.7772
#>  -1.2216 -2.7229  1.6786 -2.1235  0.4532
#>  -0.7620 -0.2503  4.3197 -0.7740 -1.8263
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
