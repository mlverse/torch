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
#> Columns 1 to 8  -3.4770   3.2509 -10.3038  -2.4920  -7.2988  -3.4729 -10.4540   7.4222
#>    2.7828   4.4454  -3.4450  -3.9025 -11.5890   6.7053  -1.3767  -4.9096
#>    6.2131  -8.3110  -5.1559   3.1771  -4.4117   8.6462  -7.9214 -13.6363
#>    1.5833  -6.9790  -2.3193  -7.1262   2.0902  -1.9993   9.5416   5.7129
#>   -0.8179  -4.0317  11.5905   5.6077   6.0992  -1.0757   3.9988 -12.0262
#>    1.8736   7.9746 -12.9472   0.8545  -8.8722  -5.0611 -14.5049 -14.7277
#>    2.4410 -11.1549 -11.4636  -7.7314   5.3260   8.3720  -1.8563 -12.1387
#>   -2.7874   0.1833   1.3043   8.1333 -18.9951  12.7389  -2.7665   9.3595
#>   -0.5399  -0.9824  -6.2963  -5.2559   7.1467   9.6196   3.0951   3.1445
#>   -4.7234   4.8380  -2.9188   8.0640   6.3623   5.2584 -10.6310   3.9458
#>   -6.3574   5.5095   4.3350   1.0935   5.7146  -6.5822   5.6480  -4.0892
#>    1.8932  -8.7734  -0.0446 -11.9076  -1.9814   0.5968  17.1492  11.5566
#>   10.1653  -8.3257  10.2897  -8.1318  -4.9406  15.0416   8.2748  18.0654
#>   -2.4693 -10.8888  -3.0984  -3.7022  10.1774  -7.3312  12.6554   2.4222
#>   -0.6143  -2.6997  -1.3709  -6.4309  -5.0475  -7.5346 -13.7060   2.6348
#>    0.0354  -7.0367   3.3425  10.3428  12.5892   6.8984   6.4173  10.0659
#>   -0.0765   4.5452  14.3068   1.8882 -10.6938  -8.5058   3.3992   3.7798
#>    1.5822  -9.3844  -0.1980 -16.3276  -2.4324 -17.5364 -15.3464   1.4345
#>   -7.4325  -0.3105   1.3748   0.2152  -1.9289   2.2583  -3.3424  -5.7505
#>   -4.6015   5.7482   0.5063  -7.6633  12.4984 -15.0608   3.5303 -12.9331
#>   -0.0441   9.3426  -8.2338  10.0529   1.7843  -1.1896  -7.1588  -7.0313
#>    4.7215   9.6653   7.2074 -15.3299  -0.9792   3.6772   3.3117 -20.4066
#>    8.7746   4.8462  -1.7559 -10.0135  -6.5929  -7.5336   1.8993  11.2685
#>    3.5494   0.5708   4.6591   4.1561   7.9955  -6.1047  -8.2184   7.6629
#>    5.5013  -9.2507   0.4163   1.7928   9.5197  -2.8035  -3.2101 -10.0926
#>    1.3088   2.3359   0.8336  -3.4400  11.7129   4.4494   8.3276  10.0126
#>   -0.9665 -10.3740  -5.4725 -19.2971   9.1550  -2.7640  -1.7740 -21.6874
#>    1.2217  -6.0522   0.2355  -7.3305   2.9779   2.4087 -12.9221  -9.0772
#>   -3.1791  -9.8706   3.6936  -1.5886  19.5121 -12.6391  -4.7239  -0.6068
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
