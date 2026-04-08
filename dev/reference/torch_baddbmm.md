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
#>  0.9797 -0.7856  3.4292  0.3511  0.0211
#>  -1.1341 -1.7491  0.3404  1.0619 -0.2565
#>  -0.2476 -1.4233  3.0678  1.5018 -1.4715
#> 
#> (2,.,.) = 
#> -1.9098 -0.5083  3.4101 -3.5053  1.6025
#>  -1.1294 -2.1521  0.4326 -1.0480 -3.4968
#>  -3.2628 -1.2406  0.0611  2.1088 -0.7255
#> 
#> (3,.,.) = 
#>  3.5882 -1.2648  0.2263  0.5318  0.3728
#>   2.1651  0.2170  0.3443  1.7921  0.1716
#>  -0.4026  1.9230 -0.4242  0.9362  0.7095
#> 
#> (4,.,.) = 
#> -1.2465  0.1044 -2.8413  2.2557  1.1593
#>  -0.7894 -0.1551 -3.7164 -0.4650 -0.4449
#>  -1.9879  2.3584  0.3822  2.1848  0.2135
#> 
#> (5,.,.) = 
#> -0.1350 -1.5099  2.1379 -2.9249  2.9627
#>  -0.7039  0.7753 -2.4279 -2.1703 -1.3975
#>   1.0913 -0.1359 -2.2453  3.0712 -3.1385
#> 
#> (6,.,.) = 
#> -2.2454 -0.5310  3.5287  1.5333  0.3425
#>  -0.5204 -0.1338 -0.1363  0.0985 -1.6340
#>   1.4309 -0.1491  1.6852  1.6957  2.0827
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
