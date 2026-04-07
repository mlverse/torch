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
#>  1.0876  0.1008  1.2359  4.0856 -2.8046
#>  -1.4988 -0.5928  0.1266 -4.8028  0.2511
#>  -1.6046  0.8232  1.2740  3.7337 -0.7596
#> 
#> (2,.,.) = 
#> -1.4268  1.9288  2.0255 -2.5394  1.0832
#>  -1.5775 -0.4890  1.4090 -2.3411  0.8797
#>  -0.9481  2.3222  0.5128  0.0432  0.1763
#> 
#> (3,.,.) = 
#> -2.7869  0.8579  2.8505 -3.0771  1.8260
#>  -1.7661  0.5416  0.4770 -3.3438  1.8213
#>  -0.4265 -0.2478 -1.5156 -0.8628 -2.2886
#> 
#> (4,.,.) = 
#> -0.9096 -1.4679  1.3980 -0.5553  2.2329
#>   0.8765 -1.7756  0.9952 -1.1754  0.5493
#>   0.2092  1.2479 -0.0793  2.9593  0.9863
#> 
#> (5,.,.) = 
#> -0.7785 -0.5610 -0.2313  0.9063  0.5250
#>  -0.6107  3.2734 -3.5440 -5.7928 -2.2906
#>  -2.2172  2.2920 -1.8258  1.3230  1.5699
#> 
#> (6,.,.) = 
#> -0.2151 -3.1240  0.1245 -0.0858  1.6454
#>   1.6317  0.6600  1.6238  0.7174 -1.5039
#>  -0.5305 -1.8137 -1.1179  0.4166  0.7154
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
