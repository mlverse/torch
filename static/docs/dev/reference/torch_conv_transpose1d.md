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
#> Columns 1 to 8   3.4433 -12.5615  -7.5613  16.0897 -11.2316   3.2326  12.9378 -12.0570
#>   -3.6381  -0.1402   0.4949  11.7678   8.9159  -8.4482 -14.6391  -0.7939
#>    2.5884   6.7424   2.8390  10.2126   9.4997 -16.1008  -5.5833  12.3043
#>    2.6724  -0.7248 -12.8631 -18.3774   3.8067  -8.2181 -13.7185  -2.7852
#>   10.1922  -2.6252  -2.3332  -9.3339  -4.1072  10.7940  -3.5652  27.0130
#>   -0.1722   3.0702   6.2452  -0.0049  10.7874  -0.7237   5.4325  -2.1277
#>   -5.1806   5.0118   1.6182   4.9711  -9.4441  11.0838  10.2300  -2.8583
#>   -6.1987   4.6459  -1.5310   4.7217  -6.8788  -1.1705 -12.8641  -9.0563
#>    4.3294  -3.3699   1.8846 -10.1660  -8.4101  26.2456   4.7961  -0.9809
#>    2.0243   6.2432  -8.7975   1.8576   1.6999  -5.4558   5.0610  10.4591
#>   -4.3427  -3.3507  -0.5977  -1.5626   6.5500  11.0638  -7.9774  -1.0043
#>   11.6796   9.1749   4.1558  -0.5090   1.0616  -0.5652   1.5407  -7.9757
#>    1.9764  -0.3825   0.3415  18.0622  -4.7380  -7.5602  16.1517   4.6455
#>   11.8046  -3.3408   3.9913   1.7897  -6.9075  -8.0261   0.1388   2.5178
#>    7.3777   8.7829   3.4677  -2.3937   6.7706  14.2669  -8.2161  10.4208
#>   -4.5376  -1.3447  -5.8222   4.3550 -10.4447  -3.3166   4.4321  -8.9870
#>   -6.8056 -10.1380   2.4376   2.0698  -0.8979 -26.0126  -2.2649  -3.7081
#>   -4.0712  -2.7415  -9.0105 -13.1834   7.5265   9.4097  -2.4883   8.0209
#>   -4.4287  -6.8022  -4.5348  -0.1508  -1.8706   3.9658   7.1839   0.4033
#>    3.4764   0.6929   6.0635  13.4752   3.8198  12.4355  -7.9987  -1.2944
#>   -3.1263  -7.1323 -10.5191   9.3957   2.8963 -10.5102   8.0877 -11.0192
#>    7.7051   9.7937  -5.2977 -11.8240  -0.2171  -9.2146  -8.1988   0.3596
#>    0.2307  13.4729  -4.3101  -4.2593   3.8784  -3.0270   3.3227  -2.9167
#>    4.1978  -4.8004   5.3049   4.9067   5.8982   6.7560 -16.3166   9.7011
#>    3.6603   3.4619   0.4659   4.2147  -0.0122   1.0212  18.2157  -8.0449
#>    7.3514   5.2970  -2.6872  15.0773   0.2837  15.4515  -1.7167  10.4609
#>    0.7517  -0.0071   3.8179  -7.4867   3.2100   4.5112  -9.4788   2.9845
#>   -7.9516   9.7015  11.7560  -4.8896  -4.4227  -3.6687 -23.2280 -13.9926
#>   -6.1424 -21.9110  -5.4683   4.0229  -0.6993  -9.6609  -5.7036  -5.8840
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
