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
#> -0.4966  1.6383 -0.0884  0.7040 -0.6964
#>   0.8204  3.2588 -1.3995  2.7592 -2.3986
#>   4.0376  0.7780  0.7175 -1.2225  2.9484
#> 
#> (2,.,.) = 
#> -2.8199  0.3877 -2.5916 -0.0165  0.8014
#>   0.0927  1.6664 -1.8840  0.9667 -0.2105
#>   2.3519 -3.7289  2.3455 -1.9142  1.1219
#> 
#> (3,.,.) = 
#> -0.2767  0.3140  2.8227  4.9703 -0.6015
#>  -0.8500  0.1478 -4.6286 -0.9227 -0.5075
#>   0.6297  1.2337 -1.9959  1.7804 -0.8562
#> 
#> (4,.,.) = 
#>  0.5680 -2.1370 -1.9033  0.9478  3.1234
#>  -1.4541  0.3124  1.5323  1.9653 -2.7419
#>   0.9565 -1.9076 -2.7929  0.1735  3.2368
#> 
#> (5,.,.) = 
#>  0.6945 -0.8881  0.0623 -0.4566 -1.2683
#>  -0.7306 -1.0268 -1.2763 -0.4201  0.8674
#>   0.0890 -2.7211 -0.8070  0.2816  2.1671
#> 
#> (6,.,.) = 
#> -0.7460  0.1051  0.7722  0.8903  3.1062
#>   0.5469 -2.7891  3.8795  1.1021 -0.3217
#>  -0.6274  2.1893 -1.9684  0.9405  2.8042
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
