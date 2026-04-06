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
#> -1.6515  0.1881  0.7381  0.9638 -1.1704
#>   1.9684 -2.4154  0.2984 -0.6663  2.0421
#>   4.1615 -1.9048  0.6935  0.9111  2.5284
#> 
#> (2,.,.) = 
#>  1.0222  0.1767 -3.3361  1.5442 -0.4838
#>   0.7010  0.6935 -1.0112 -0.2502 -1.2109
#>   0.1288 -1.7114 -1.8729  0.2002  0.9353
#> 
#> (3,.,.) = 
#> -1.0483  0.7429  0.6723 -2.8746 -2.9312
#>   0.8534 -1.3290  3.7568 -4.7052 -2.6159
#>   1.3785 -3.3964  0.8801  2.9234  1.3314
#> 
#> (4,.,.) = 
#>  1.6650 -1.1365  0.9760  0.6719 -2.1714
#>  -0.8057  1.0819 -0.1200 -0.9780 -2.3412
#>   0.0689  0.2010 -1.6690 -0.6991 -1.5357
#> 
#> (5,.,.) = 
#>  3.0826  2.4604  1.5748  1.9015  2.9173
#>  -0.7943  1.7071  0.2674 -0.2381 -0.8105
#>   0.8056  8.9756  3.7678  0.2760  3.4203
#> 
#> (6,.,.) = 
#>  1.8071  0.0451  2.1684  3.3743  1.0931
#>  -1.5170 -0.2503  1.0120 -0.7479 -0.1122
#>  -3.2988  1.7331  0.4249 -5.0674  0.0631
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
