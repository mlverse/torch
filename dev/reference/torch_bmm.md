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
#>   0.4373  1.5634 -1.0256  0.1374 -1.4827
#>  -0.9561  0.8756 -1.1238 -1.3292  0.2638
#>  -3.1086  1.4404 -3.8571 -2.5704 -4.6629
#> 
#> (2,.,.) = 
#>   0.7073  0.2591 -0.3842 -0.9896  0.3566
#>  -1.1111  0.5941  0.1596 -1.3725 -1.5695
#>  -1.5108 -1.9175 -2.7106  8.1097  4.5661
#> 
#> (3,.,.) = 
#>   0.2165 -1.7046  0.2384  1.5517 -1.7653
#>  -7.3211 -2.4128 -1.1437 -9.2435  4.6066
#>   4.1217  1.3329  0.6167  6.8285 -1.9047
#> 
#> (4,.,.) = 
#>  -0.7813  0.4700 -1.8805 -0.4603 -0.3814
#>  -3.7203 -0.2842 -4.0710 -0.4720  1.2508
#>  -1.6257  1.1717 -0.4896 -2.1428  2.2338
#> 
#> (5,.,.) = 
#>  -3.5574 -0.6332  1.5894  0.5712  0.5916
#>   0.9357 -0.3563  0.0991  1.1161  0.2897
#>   2.4466  0.0892 -0.4531  1.6422  0.5252
#> 
#> (6,.,.) = 
#>   1.6430 -0.4931  1.0867 -3.4624 -2.9175
#>  -1.4861 -1.2549  0.4561  0.0874 -1.9583
#>   1.5040 -2.6609  3.4090 -5.3412 -7.0144
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
