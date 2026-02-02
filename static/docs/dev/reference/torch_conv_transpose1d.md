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
#>  Columns 1 to 8   6.3955   4.7005   3.7903  -3.3954   4.7633 -11.5244   4.2495  -0.9016
#>    1.3215  -4.0865   3.3701  -6.7171  -9.4137  -2.0142   5.6152 -28.2741
#>    4.7376  -0.4987  10.5346   6.7221  -2.7133   0.6075  -5.7980   0.9793
#>   -2.8519  -0.1094  -1.0312  -1.1326  -2.6600   2.7575  -1.7069   5.3676
#>   -0.5486   4.5974  -0.0872   2.1235   1.1761  -9.0643   2.8232  -4.3857
#>    2.6659   0.8100   4.6878  -6.5950  -2.3879   7.5999   9.8873  -8.7269
#>    1.0674   2.3693   6.3618  -7.0026   9.1325   2.4114  12.4715  -4.6229
#>   -1.0492  -3.5690   6.3763  -0.0517   1.1954  22.7857  10.0385  11.5630
#>   -1.4826   4.4838   3.8699  -3.5349  -0.8945   5.0012   1.3129  -7.0377
#>    1.0935  -0.2457   1.5516  -2.2440   5.8277  -4.0479 -10.3351  11.7462
#>    3.2864   1.9711   6.5403  -6.5636  -4.4206 -13.8676 -22.9147   1.2957
#>    5.1440   1.8342   4.7762   7.8626  12.7029   6.4531  -1.8055   1.8274
#>    3.3633   3.2935   5.2666   2.6624   5.9664   9.5963  -3.2458   1.5947
#>    2.0339  -1.9927   1.8332   7.5023   3.6500   0.7680   2.2852   1.0376
#>   -4.1279   3.6952  -1.1160  -4.9810   7.7477   6.8615  -8.9050   6.0454
#>   -1.7017  -3.7173 -11.3147  15.7938 -17.8386   2.1883 -12.1158   0.0887
#>    6.6693  -3.4248   0.2732  13.3540  -3.7032  -2.2156  -0.7298   0.0796
#>    0.9826  -4.2085   1.4489  -6.3101   3.6986   0.3170  -4.2850  -5.8351
#>    3.9554   3.7326  11.6760   4.9532  -1.0502  -5.1748   8.0858  -5.6361
#>   -3.5061  -1.7462  -9.2657   7.9445   9.9995 -13.0653 -13.9202   3.9183
#>   -4.6600   5.8180  11.8954  10.9781   5.9981  13.8726  -1.1746   3.0478
#>   -1.6199  -3.1444 -17.7185   9.6022 -12.4274  -4.3217  -5.1853  -0.7426
#>   -4.0012   3.1576   3.8208  -3.2495   5.6348 -14.8232  -9.0420 -12.3315
#>    2.3595   9.2152  -1.7915  18.2671  -7.1064 -29.6398 -17.5000   5.1708
#>   -3.7784   4.7356   1.9821  -3.0646 -12.0845  -3.4788   2.3457   5.4913
#>    4.6159   3.9093  13.4059  -6.0717  -2.5717   2.6315   1.1614   8.9079
#>   -1.5215   2.9828  -6.1992  -9.0892  -1.1164   1.8031 -20.9351   1.9607
#>    2.4306  -5.2401  -2.6845   1.2330   0.1274  -8.0022   0.7998   8.6060
#>  -10.7621  -1.4020  -6.9855  -2.5610   7.6361   0.0636 -14.0231   1.4129
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
