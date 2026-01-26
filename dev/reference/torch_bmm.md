# Bmm

Bmm

## Usage

``` r
torch_bmm(self, mat2)
```

## Arguments

- self:

  (Tensor) the first batch of matrices to be multiplied

- mat2:

  (Tensor) the second batch of matrices to be multiplied

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
#>  -0.2066  0.6858 -1.1152  1.7695 -1.7994
#>  -0.2213 -0.9372  2.3052 -0.7337  4.4298
#>   0.6470  3.9441 -0.4010  0.2195 -0.7622
#> 
#> (2,.,.) = 
#>   0.9979  3.0425 -2.9398 -0.4140 -0.7378
#>  -0.1309  3.6280 -3.1569 -0.8569 -2.4318
#>   0.6820  0.0826  0.1122  1.8858  0.7388
#> 
#> (3,.,.) = 
#>  -2.3701 -0.8235 -0.0524 -0.0260  0.5170
#>   3.2244 -1.8397  4.6548 -1.2347 -1.4527
#>  -1.8660  0.2253 -3.6784  1.1478  1.0806
#> 
#> (4,.,.) = 
#>  -3.3281 -2.9369 -0.9400 -1.2274 -1.5185
#>  -2.0421 -1.0693 -2.0042 -1.3740  0.4938
#>  -3.0910  3.3647 -2.1134 -0.9174  2.1711
#> 
#> (5,.,.) = 
#>  -0.1146 -0.0108  0.2453 -0.9772  0.9086
#>   0.3050  0.0130  1.2932 -0.3329 -1.4111
#>  -0.3479  0.1176 -2.0135 -0.3096  2.0420
#> 
#> (6,.,.) = 
#>   2.8133 -0.0916 -0.0908  1.2120  1.8348
#>  -0.8835 -0.3492  0.5797 -0.5200 -0.1657
#>  -2.6346 -0.2039  1.7446 -0.2154 -0.6298
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
