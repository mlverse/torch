# Bmm

Bmm

## Usage

``` r
torch_bmm(self, mat2, out_dtype)
```

## Arguments

- self:

  (Tensor) the first batch of matrices to be multiplied

- mat2:

  (Tensor) the second batch of matrices to be multiplied

- out_dtype:

  (torch_dtype, optional) the output dtype

## Note

This function does not broadcast . For broadcasting matrix products, see
[`torch_matmul`](https://torch.mlverse.org/docs/dev/reference/torch_matmul.md).

## bmm(input, mat2, out=NULL) -\> Tensor

Performs a batch matrix-matrix product of matrices stored in `input` and
`mat2`.

`input` and `mat2` must be 3-D tensors each containing the same number
of matrices.

If `input` is a \\(b \times n \times m)\\ tensor, `mat2` is a \\(b
\times m \times p)\\ tensor, `out` will be a \\(b \times n \times p)\\
tensor.

\$\$ \mbox{out}\_i = \mbox{input}\_i \mathbin{@} \mbox{mat2}\_i \$\$

## Examples

``` r
if (torch_is_installed()) {

input = torch_randn(c(10, 3, 4))
mat2 = torch_randn(c(10, 4, 5))
res = torch_bmm(input, mat2)
res
}
#> torch_tensor
#> (1,.,.) = 
#>  0.6746 -2.2887  5.1242  0.1473 -2.6737
#>   3.0315  0.5688 -4.2874  0.8894  0.3080
#>   0.0343  1.5444  2.3396  0.9668 -2.7272
#> 
#> (2,.,.) = 
#>  3.9741  4.1420 -1.2975  2.2054  1.2925
#>   2.2339 -3.8832 -2.1716 -1.1528  0.4034
#>  -1.4692 -0.5160 -0.8785  0.8101 -1.2997
#> 
#> (3,.,.) = 
#>  1.6220  1.0653 -0.0169  0.5721  0.3522
#>  -1.1270 -0.5914 -0.5405 -0.9430  1.1117
#>  -0.8535 -1.6479 -2.2086 -2.4816  3.0231
#> 
#> (4,.,.) = 
#>  0.8184  0.5700 -2.0259 -1.0429  0.1848
#>  -2.7500  4.0096  0.0670  3.8473  1.0122
#>  -2.1363  2.9846 -2.2307  1.5210  0.7500
#> 
#> (5,.,.) = 
#> -0.3835 -2.7102 -4.0611  1.6586 -2.8288
#>   0.0499 -2.5442  0.5702  1.7817  0.5564
#>   1.1550 -2.1463  0.7236  1.1917  1.3632
#> 
#> (6,.,.) = 
#>  0.6165 -5.6784 -3.9762 -0.2252  1.2589
#>   0.4797  0.4270 -0.2013 -0.8507 -1.1556
#>   0.2726  1.6412  1.7531  0.9424  1.5135
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
