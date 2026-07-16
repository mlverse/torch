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
#>  2.2000 -0.7923  0.2935  0.4767 -1.2085
#>   0.8095  1.8000 -1.7164  0.5610 -2.7155
#>   1.2809 -4.8065  1.7066 -2.3588 -1.1120
#> 
#> (2,.,.) = 
#>  0.6404  1.1182  0.7750  0.9381 -1.0062
#>   0.6219  0.0073 -0.8583  1.1413 -0.0003
#>  -3.1224  1.4612 -1.8890 -0.0049 -1.5393
#> 
#> (3,.,.) = 
#>  0.2863 -1.8554  0.4992  0.0138 -0.4216
#>   0.8800 -1.6296 -1.1771 -0.5292  1.0614
#>  -0.5872  0.2990  0.0820 -0.2509  1.5441
#> 
#> (4,.,.) = 
#>  0.7325 -1.4284 -1.0431  4.8532  0.4547
#>  -2.5459  1.8793  1.0291 -0.7198 -1.2232
#>  -4.5739  0.8963  2.5107 -0.6444  0.6756
#> 
#> (5,.,.) = 
#> -0.3260  1.2799 -0.0002  0.4879  1.4394
#>  -0.5184 -0.6462 -1.0720  0.8236  0.5054
#>  -1.8312 -1.9104 -2.4760  1.0984  1.3151
#> 
#> (6,.,.) = 
#> -1.7282 -0.7739  1.6819 -1.9606  1.0053
#>  -2.3394  0.6512 -0.6166 -2.3904 -0.1609
#>  -0.1269 -0.1459 -1.3086  5.2988  0.6356
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
