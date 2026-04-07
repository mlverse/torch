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
#>  0.7644  0.2749  0.3943  0.4731  0.6419
#>  -1.2109 -0.5306 -0.2910 -0.5461 -1.3319
#>   1.8080  0.0722 -2.3320 -1.0215 -0.6572
#> 
#> (2,.,.) = 
#> -0.8793  0.4138  1.0695 -0.5361 -0.3516
#>  -0.2679  0.1485  2.2845  0.0379  0.2810
#>  -0.5867 -0.4545 -1.3071 -0.5426 -1.2081
#> 
#> (3,.,.) = 
#> -2.4876  0.5725 -3.5099  0.9187  0.6029
#>  -1.8374  1.8161 -2.5323  0.3243  0.2712
#>  -3.6351  6.0541 -1.7580  0.6852 -0.9298
#> 
#> (4,.,.) = 
#> -0.5673 -6.5445  6.3770 -4.4982  4.1872
#>  -1.0938  1.7172 -0.1429  0.5941 -2.4658
#>   1.2005  2.6544 -3.0743  2.1837 -1.3696
#> 
#> (5,.,.) = 
#>  3.2517 -3.5553 -5.2178 -3.3571  3.9776
#>  -0.0259 -0.4085  4.6550  0.6120 -0.0620
#>   1.6255  0.2472  0.3880 -2.5581  0.9220
#> 
#> (6,.,.) = 
#> -3.6326  1.4060 -2.0844  1.2913  0.8505
#>   1.5001 -0.2832  1.8095  0.1948 -1.7140
#>   1.0012 -0.2249 -1.1077 -1.5385  2.1662
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
