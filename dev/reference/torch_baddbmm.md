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
#>  -2.0059   0.7142  -3.8191  -1.5877  -2.0410
#>    2.7436  -1.1614   1.7112  -0.2320  -1.4822
#>    0.9231 -10.3405   6.4463  -0.7598   1.8548
#> 
#> (2,.,.) = 
#>  2.2567  4.2367 -1.2973  7.2875 -4.4606
#>   2.8565  1.8022  1.3385  3.1771  1.6353
#>   1.1334  0.6159  4.1339  2.8531  2.6251
#> 
#> (3,.,.) = 
#> -0.1688 -0.5647 -0.4082  1.2599 -1.5071
#>   0.3505  0.4195  1.0703 -0.3364  2.1057
#>   1.0065 -0.9794  2.2630  3.8912  0.1977
#> 
#> (4,.,.) = 
#>  4.2344 -0.7462  1.3021  2.8666  1.5442
#>   2.3662 -0.2567 -0.7632  1.3044  0.2818
#>  -3.4545 -0.8377  1.6532 -0.6032  1.2000
#> 
#> (5,.,.) = 
#> -0.6707  0.7084 -0.0999 -0.6491 -1.4388
#>   0.8721  4.9825  0.9353  3.8310 -1.1523
#>   3.0727 -3.9191  2.8944 -2.2201 -2.7476
#> 
#> (6,.,.) = 
#>  1.0143 -1.7559 -0.2836  0.0458 -2.1289
#>   2.7856  1.8777  2.3884 -1.2292  3.1660
#>   0.0614 -4.1004 -0.5260 -2.0018 -3.3490
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
