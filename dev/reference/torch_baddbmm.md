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
#>  2.9291 -1.0895 -0.9465 -3.8789 -0.3461
#>  -0.5643 -0.5178 -1.9799 -1.1077 -1.4207
#>   3.0915 -0.9116  0.9357 -3.6362  1.2873
#> 
#> (2,.,.) = 
#>  0.7347  1.6655 -0.7775 -0.9839  0.9941
#>  -5.5155 -4.3208  2.2989 -1.5198 -0.1937
#>  -1.1706 -3.2572  1.0095  2.2166 -2.8943
#> 
#> (3,.,.) = 
#> -1.8155 -0.6057  1.9485 -1.1498 -0.6451
#>  -0.1863  1.2720  1.6506 -0.5624 -1.0127
#>  -4.5338  1.5370  1.9342  1.2794 -5.8346
#> 
#> (4,.,.) = 
#> -0.5217  0.6356  2.9203 -3.4558  3.4173
#>  -0.8244  0.4059 -1.0766 -3.9337  0.5773
#>   1.7444 -1.9841 -0.6976 -1.7848  1.0965
#> 
#> (5,.,.) = 
#>  1.6923  1.4848  1.4066  4.8408 -0.3392
#>   0.4011 -1.1269 -1.9445 -2.6581 -0.7644
#>  -0.4992 -0.4644  2.4490  1.6286  0.8118
#> 
#> (6,.,.) = 
#>  1.2415 -7.3667  1.9358 -1.5171 -1.0562
#>  -0.5222  3.0083  1.8535 -0.2571 -0.3018
#>   1.2493 -2.4618 -1.4588 -0.6141  1.9486
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
