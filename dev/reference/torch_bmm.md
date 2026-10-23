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
#> -2.0844  2.1096  1.7651  1.3897  0.4385
#>   2.1741 -0.6303 -0.4057 -0.1807 -0.8078
#>  -2.4256  0.4978 -1.1268  0.4111  2.4453
#> 
#> (2,.,.) = 
#>  0.5166 -2.6437  3.3382  1.4287 -1.1268
#>   2.1850  4.2945 -3.2428 -1.3477  2.0430
#>   0.7069 -0.1335  3.1446  0.4192 -1.0401
#> 
#> (3,.,.) = 
#>  0.4940 -3.0488  3.8309 -1.3729 -2.7261
#>   1.0882 -0.0630 -1.3179 -0.4433 -1.4485
#>  -0.3537  1.3796  1.8840  2.4117 -1.9630
#> 
#> (4,.,.) = 
#>  0.8513  1.7387 -4.8825 -0.4613  1.1020
#>  -0.7287  0.4940 -4.2167  1.5403 -1.1335
#>  -1.0927  1.8368 -3.0823  0.4023 -0.5621
#> 
#> (5,.,.) = 
#>  0.6539 -1.6827 -0.2331  0.9477 -1.0577
#>  -0.6274 -1.7364  1.1672  0.0776  2.2178
#>   0.1177  0.7135  2.2523 -0.4889  1.2056
#> 
#> (6,.,.) = 
#> -0.5484 -2.7654  3.3790  1.9337 -0.7445
#>  -0.9357  1.8112 -2.1044 -0.6504 -0.6670
#>  -2.4254 -1.2622  1.3638  5.9520 -1.2103
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
