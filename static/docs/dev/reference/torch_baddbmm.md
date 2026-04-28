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
#>  5.5877 -2.3177  3.6544 -1.2013  1.1242
#>  -8.0347  2.0900  2.5432  0.3836 -2.7820
#>   1.5498 -3.7470 -4.9541  3.0016 -0.7278
#> 
#> (2,.,.) = 
#>  1.8160 -0.8787 -1.2474  3.0191 -0.8595
#>   2.1906 -2.1126 -4.0500 -0.8549  2.3741
#>  -0.4082  0.2575  1.4023 -1.8565  1.1616
#> 
#> (3,.,.) = 
#> -1.3373 -0.1667  1.7885  1.4854 -0.0068
#>   1.0683  0.8124  0.2660  0.3066  0.6935
#>  -3.2506  1.8222 -0.5092  1.4676  0.1320
#> 
#> (4,.,.) = 
#> -1.6023 -1.4500 -0.4413 -0.9261  0.0545
#>  -0.2585  0.7005  0.9554  3.0534 -0.7866
#>   0.4781  0.3474  2.6123  0.1430 -0.5234
#> 
#> (5,.,.) = 
#> -1.9827 -0.5562  2.8691 -2.2620  0.9605
#>  -0.2619 -1.9832 -0.1109 -3.1390  1.2579
#>   0.5128  0.1501  0.2079 -0.4940  1.7543
#> 
#> (6,.,.) = 
#> -0.7291 -1.1605 -2.7308 -0.4762  0.1426
#>   2.5679  3.0251  3.2035  1.3237 -2.1193
#>   1.0792 -0.1212  0.1610  0.9200  0.7137
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
