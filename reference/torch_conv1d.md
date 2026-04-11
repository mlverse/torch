# Conv1d

Conv1d

## Usage

``` r
torch_conv1d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a
  one-element tuple `(sW,)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a one-element tuple `(padW,)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv1d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See
[`nn_conv1d()`](https://torch.mlverse.org/docs/reference/nn_conv1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

filters = torch_randn(c(33, 16, 3))
inputs = torch_randn(c(20, 16, 50))
nnf_conv1d(inputs, filters)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 8  10.1366 -12.5879  -2.3987   2.1057   1.2087   9.1452  -6.3392   4.5060
#>    6.6529   5.6660  -8.3291   3.8192   1.5280   5.3013  -1.8791  -1.6807
#>    4.5772  -4.7618  -7.0345  -1.6540  -4.2512   7.7169  -5.0014  11.1350
#>    7.3237   8.3005  12.6702   1.6081  -8.3938  -0.2422   0.8141   5.8440
#>   -6.8993  -2.3302   5.4941   3.7813   4.4414  -1.3626  -3.1319  -0.5180
#>    4.7533   2.7238   1.7546   3.8819  -1.9405  -3.9507   0.7671   0.4220
#>   -2.7296  -0.2756   1.4673 -13.2371  -8.3785   2.0706  -2.1222   3.9485
#>   -0.8368  -1.6146  -8.2214   2.7421   4.3445   0.7194  -6.9788   0.2724
#>    5.9263  -1.6689  -1.4666  -2.9054   2.1946   4.1590   5.5060  -3.3223
#>   -1.8797  -1.8989   3.4670  -1.5641  10.5341   3.1161  -8.2537  -8.7843
#>    1.4935  -1.6778   6.1705   5.4464  -2.7905   1.0212  10.1735   8.7304
#>    3.8516  -2.4424   0.0366  16.5566   6.6281   0.5291   1.0062   4.7484
#>   -0.3120  -3.7671   0.3188  -0.7345  -0.6022   2.6879  -1.2977   1.6850
#>   -1.1293  -2.3883  -5.0961   4.9866  -7.5344   0.8650  -3.6437   7.2965
#>   16.8842  11.9622   3.4974   8.8972   4.6152   4.9219  -4.2298  -0.1799
#>   -3.6625 -14.4415   3.4079   3.7139  -7.0021 -14.4776  -6.9672  -4.8209
#>   11.1947   1.2507   3.7756  -0.2670  -8.3802  12.9284  -6.0678   1.7588
#>   -5.3913  -8.8218   3.0949   5.7897  -6.3966  -2.9199   6.8805  13.6124
#>   -4.0431  -2.0094   2.3324 -12.8273   8.1435  -6.7683   2.2129   2.8533
#>    1.9120  15.5639  -1.6043 -18.2548   7.6353  -3.5180  -2.3774  -7.2835
#>    2.3384  -0.2035  -2.6773  -2.0556  -0.6385  -2.2843   7.8393  -5.0591
#>   14.6307  -4.7248  -3.9032  -9.7979  -2.7705  -7.8486   7.6669  -8.0510
#>   -4.6492   1.2951  -2.7158   6.3768   1.4562  -5.6259  -1.1428   0.4762
#>   -2.8530   3.2673  -8.8869  -5.2970   9.8446  -8.2838   1.9769  -1.3380
#>  -12.1557  -8.1958  -1.0436   0.0499   7.4135   4.8176  -5.4264   1.2302
#>   -0.5709  -2.6675 -10.5426 -19.2830  -5.0179  -6.7538   0.1035  -9.6744
#>   -0.4326  -3.7613  -1.1358  -7.2505   0.2943  -9.2221   5.4841  -3.4903
#>   -5.0554  13.1854  -6.7358  -6.0664  17.3466 -10.4646   0.8322  -3.0264
#>   -2.3564  -7.6455  -8.4678   5.4166  -6.2224  -4.5906   5.9022   2.9745
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
