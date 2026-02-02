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
[`nn_conv1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv1d.md)
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
#>  Columns 1 to 8  -9.1113   2.6939 -10.4815   6.5575   3.4934   5.6334  -0.8486   6.5031
#>   -6.3609 -12.0871   7.2208  -4.7838   4.7878  10.7295  10.2849  -1.3008
#>    5.9428   2.6709  -5.8642  -7.0827  -3.2397 -10.5248  -8.1156  -0.7790
#>   -7.3410  -0.5296   2.0827   0.0251   4.3511   8.4664   0.0552  -3.7068
#>   -7.1754   8.2712   8.7984   3.7731   4.6843  12.2015  -7.0274  -1.4482
#>   -6.0479  -1.9389   0.1458   5.3020 -11.1864  -0.1259  11.8151  -2.1028
#>   -3.9772   7.0192  -8.0536  -7.5005   1.3787  -1.7771  -3.4619  11.6646
#>  -15.4079  -9.2418   0.7770   7.3457  -5.7920  -3.7961   3.4830  -2.2469
#>   16.4063   9.5252  -2.7002  -3.0561  -4.0890 -12.9830  -0.0770   2.3610
#>   -8.1272 -11.0811  -7.1750   3.5053  10.4166  13.7875   6.1878   3.9509
#>   -0.7356   2.8585  -0.2559   3.5441   4.1840  -2.1609  -1.7722   4.4445
#>    7.2017  -6.9091   7.6663 -16.4230 -10.2100   1.7795  -4.5030   1.8963
#>  -10.8028  -1.1207   1.2729  -8.9145  10.5911   1.3278  -7.4381   2.3106
#>    6.7011   6.7979  -6.9180  16.2756   2.4357  -3.3821   7.5742  -2.8106
#>   -6.4210  -6.5394   3.0428   0.1903  -3.7371  -1.9731  -2.7985  -3.0673
#>    9.6035  -1.3861   5.8473   4.4234 -14.9022   1.2609   6.4570  -1.5714
#>    2.7564  -5.7539  12.0096  -8.5126   2.3222  10.6029  -1.6954 -10.2475
#>   -2.3005   4.3118  12.9056   6.4700  -2.6437  -7.6435  -4.3302   0.1141
#>   11.6490  14.0180   1.4856  -1.2870  -5.9910   2.2572  12.7136  -3.7777
#>    2.6227   1.2219  -1.5245  -1.1332   1.7933   4.7846   0.6526   8.6493
#>   10.2254   5.6122  -2.2001   1.2488 -11.3030 -14.5644  13.3687  -2.3955
#>   13.0497  -1.2552  -7.4512   5.7288   3.7069   2.9595  -0.3486   4.7354
#>   -5.3367  -0.1981   8.4797  -3.7844  -6.2536   8.4050  -2.6270   9.2499
#>    4.5483  -0.4948   1.7959  -8.2335  -1.4134  17.3514   0.6226 -11.3503
#>   -1.4688  -2.8841  -1.0063  -1.7289   0.0362   5.0620   5.6923  21.0929
#>   -9.1102 -11.0280   2.8216  -0.8929  -8.9494   9.2158 -14.9600   3.2902
#>   19.0334  10.7373  -3.9551  15.0589 -12.0717 -19.6584  -1.6580   0.5051
#>   -0.0688   5.2326  -2.1909   7.0688  -2.5535   6.6009  10.9295   3.9939
#>    5.5086  -2.2544  -0.4344  10.1484   2.5947  -7.5724   7.0229  -2.2995
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
