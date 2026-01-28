# Conv_transpose1d

Conv_transpose1d

## Usage

``` r
torch_conv_transpose1d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  output_padding = 0L,
  groups = 1L,
  dilation = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sW,)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padW,)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padW)`. Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dW,)`. Default: 1

## conv_transpose1d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

inputs = torch_randn(c(20, 16, 50))
weights = torch_randn(c(16, 33, 5))
nnf_conv_transpose1d(inputs, weights)
}
#> torch_tensor
#> (1,.,.) = 
#>  Columns 1 to 8  -0.8996   2.6296 -11.1595  10.1301   5.7301   5.1652  12.7988  -8.4432
#>   -0.0681   3.7444   0.4859  -8.7046  -0.6014   3.2950 -13.2765  11.0148
#>    6.4708   1.2848  -4.3305 -11.6178   3.3857   7.8927   6.8233 -22.5834
#>   -3.2608  -4.4836  -3.2565   2.2789 -11.1877  -8.7253 -18.7369   6.7277
#>   -3.6290   2.3646   8.0246  19.2487   1.3787  -8.4757  -5.5343   3.7508
#>   -1.9140  -3.7509   8.3034   1.7739  11.9995   5.1494  -5.6747  18.3705
#>   -7.8554   1.6970   6.4104  -2.5872  10.4635 -20.2539   2.0344  -4.0191
#>    2.8359  -0.5811  -0.7254   7.5018  -4.0804  -7.9599  -0.2148  12.2950
#>    0.6915  -3.9595  -0.9240  -1.0130  14.8549   3.2660  -9.0195  -5.2064
#>    0.7127  -0.5842 -11.1161   7.5832   5.8419 -11.9828  10.5020  -1.1464
#>    5.7465   9.5822  -0.5626   4.8551 -12.0691  -8.0546  -9.0644   0.6896
#>    5.4419  -4.6399 -14.6868  12.9322  -7.7161   6.9936  -1.7416   1.2692
#>    2.2553  -0.9718   0.2957   5.0066  15.9072  -0.1481 -21.4876  -6.2407
#>    0.0608   7.6391  -2.5021  -8.6753  -1.3838  -1.8635  -5.5697   3.4675
#>    7.5981  -4.9290  -0.9624   4.2076  -8.4218   0.6679  -6.3817  13.6305
#>   -6.5646   3.9596   7.0864   9.7789  -5.1039   4.7100  -9.5419   4.9465
#>   -1.0352   0.3153   6.0907   1.8537  12.5984  -4.8836  -0.6326 -11.9576
#>   -0.2482   5.2599   3.5847  12.6145   5.5152 -10.5347  -9.3558   3.3055
#>   -1.5747   2.9176   5.4558   4.9926   3.2439  -5.4547 -14.6043  13.4074
#>   -1.8735   1.4659  10.0329  -3.2988  13.3594  -8.9342 -11.9548   7.1772
#>   -4.5271   3.3389   2.2955   0.9875  -4.5615   3.1427   7.6644  -2.3051
#>   -2.2213   2.7080  -0.3209   1.3979  -0.4220   6.3358   2.8455  -1.1382
#>    7.5433  -1.5620  -0.7239  -6.5375   7.8269  -6.2844  -5.4918   6.9408
#>    3.7707   4.3072   2.5953  -6.3165  -7.1478  -4.3195   5.2933 -12.2193
#>    4.9546   4.2081  -0.4651  -6.7228  -2.1100  -3.9166   1.6746  -5.1457
#>   -0.0848 -10.9786   8.5078  -1.3957 -14.2463  -4.7814  12.7668  -1.6164
#>    5.6715  -0.6339   3.9102  -3.0337  -2.4955   0.8257   6.3232  20.5198
#>    2.1496   7.1672  10.8184   0.8038  -9.0536  -9.7530 -15.4603  10.5456
#>   -2.6140   7.1502   0.3864 -15.2176  -0.1566  10.8256  10.1064  -8.0298
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
