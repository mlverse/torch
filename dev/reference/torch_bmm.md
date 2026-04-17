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
#>  1.7467 -0.8819 -2.7191  0.6865 -0.9532
#>  -0.0149  0.2314 -0.0970  0.2685 -3.0268
#>  -0.0094 -2.4659 -1.2616 -1.2565  2.4741
#> 
#> (2,.,.) = 
#> -0.9164  0.4742 -0.1877  0.8684  0.2946
#>  -4.0454  1.3130  0.3555  1.1375  1.2932
#>  -2.7846  0.9407  3.8317 -2.2902  1.1892
#> 
#> (3,.,.) = 
#> -2.6054  2.6795  0.2951 -0.4487 -3.4215
#>  -3.9210  1.5469  5.3679  0.5818  0.1636
#>  -1.3224 -0.1053  1.8592 -0.1941 -0.7801
#> 
#> (4,.,.) = 
#> -1.3167 -3.7156 -2.7231  0.1012  0.0949
#>  -0.6516  2.6850 -1.7121  1.4751 -2.5067
#>  -3.5432 -3.2098 -5.2698  0.6617 -1.2632
#> 
#> (5,.,.) = 
#> -0.9484  0.8092  0.9702  2.6477  1.0683
#>   0.8528  0.6634  0.0053  2.1482 -1.2180
#>   1.4792 -0.5833 -1.3974 -2.4229  1.8858
#> 
#> (6,.,.) = 
#>  0.5208  3.3174  1.9202 -0.2193 -0.1982
#>   2.4314 -5.3379 -3.2763 -1.2681  3.1623
#>   0.7208 -3.2147 -2.4782  0.5057  1.9479
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
