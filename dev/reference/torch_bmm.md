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
#>  0.3862  0.0341  1.3834  3.0581 -0.3752
#>  -0.6652 -1.2072 -0.7507 -1.2745 -0.8559
#>   0.2928  0.5080  1.1432  0.5686  0.7175
#> 
#> (2,.,.) = 
#> -0.1942  1.9242 -0.4274  1.0937 -2.9238
#>   0.2925  0.5851 -1.0353 -0.5991 -0.7563
#>  -0.8421 -5.4293 -0.7082 -1.5582  3.1982
#> 
#> (3,.,.) = 
#>  0.6886  1.4814  0.9972  0.0094 -1.2069
#>   0.8505 -1.7968 -0.8866  0.2380  0.8617
#>  -1.4581  2.2340  1.2904 -0.5346 -2.0764
#> 
#> (4,.,.) = 
#> -0.8475 -0.0460 -0.1518  1.5287  0.9254
#>   0.1307  2.7385 -1.1932  0.4262  1.1423
#>  -1.1003  1.2316 -3.4609  1.6001  1.2236
#> 
#> (5,.,.) = 
#>  0.1038  1.0975 -1.4615 -3.4055 -0.0002
#>   1.6970  3.3196 -1.2552 -2.7429  0.9389
#>  -1.1294  0.3727 -4.0900  0.4561 -0.2171
#> 
#> (6,.,.) = 
#> -2.3394  0.0489  1.6726 -3.2671 -1.7523
#>  -0.2969 -1.3431 -0.9868 -0.0243  1.2177
#>  -0.1927  1.8689  0.4172 -2.9340 -2.4100
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
