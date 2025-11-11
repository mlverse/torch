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
#>  Columns 1 to 6  2.7029e+00 -8.8443e+00  6.4511e+00  9.0348e+00  9.3440e+00 -6.7585e-01
#>  -1.8289e+00 -2.0702e+00  5.8647e+00  8.6030e+00 -1.5062e+01  4.5843e+00
#>   4.8017e+00  5.0522e+00  1.0703e+01 -6.6002e+00  1.1023e+01 -1.4135e+01
#>  -8.1586e+00 -3.8046e+00 -4.7789e-01  4.2947e+00 -4.3391e+00  1.1529e+01
#>  -2.2296e+00  8.2134e+00  9.4233e+00  1.9376e+00 -1.0596e+01  2.6929e+00
#>   2.5843e+00  7.4015e+00 -2.4028e+00 -9.6425e+00  6.5236e+00  7.2541e-01
#>  -3.8892e+00 -3.2894e+00  1.0548e+01  7.4640e+00 -6.2714e+00 -9.9469e+00
#>   2.5527e+00 -1.3737e+01 -1.2357e+01  4.7635e+00 -8.9150e+00 -1.4537e+01
#>   2.0665e+00 -7.4327e+00 -3.8717e+00  2.5725e+00 -5.5419e+00  1.9218e+01
#>   1.9737e+00 -7.7595e-01 -4.5690e+00  1.2395e+01 -9.8963e+00 -6.7317e+00
#>  -3.1309e+00  8.4824e+00  1.1321e+01  1.1142e+01  2.9009e+01 -7.1143e+00
#>  -2.1921e+00 -1.4667e+01 -2.5878e+00  9.7617e+00  4.5896e+00  8.0625e+00
#>  -2.1588e+00  3.0776e+00  4.2228e+00 -1.4731e+01 -1.0333e+01  1.0413e+00
#>   3.6961e+00  7.9374e+00  1.5108e+01 -9.6048e+00 -1.2780e+01 -7.0582e+00
#>   4.8288e+00  1.2537e+01  9.1356e+00  7.0034e+00 -8.6997e+00  1.1565e+00
#>   2.5781e+00  6.0856e+00 -8.9186e+00 -2.1429e+00  6.5063e+00  3.9911e+00
#>  -4.8896e+00 -1.0985e+01 -6.0011e+00 -1.5643e+01 -1.2591e+00  1.5076e+01
#>   2.1028e+00  9.5184e+00  6.0552e+00  7.5227e+00  6.6554e+00 -3.6983e+00
#>   4.2038e+00 -5.8750e+00 -3.8910e+00  5.0405e+00 -2.2372e+00  2.4989e+00
#>   1.0108e+00  1.1525e+00 -6.8897e+00  4.7989e-01 -5.5663e+00  6.3193e+00
#>  -8.3604e-01 -3.5239e+00  9.8900e+00  3.6694e+00  9.5051e-01  9.7126e-01
#>  -4.2962e+00 -1.7793e+00  5.1328e+00  8.7287e+00 -1.5807e+01 -6.4865e+00
#>   1.4977e+00  1.9930e+00  1.0275e+01  8.9083e+00  1.1550e+01 -1.6284e+01
#>   7.1103e-01 -5.2482e+00  1.1770e+01 -5.1838e+00  1.4927e+01  1.0633e+00
#>   2.3572e+00  1.1883e+01  5.6804e+00 -8.9215e+00 -2.4064e+00  3.1545e+00
#>   1.2588e+00  5.7127e+00  3.2865e+00  4.9583e+00  1.9027e+00  6.9967e+00
#>  -6.5220e+00 -1.4356e+01 -1.5256e+01 -2.5007e+00  1.0316e+01 -1.5478e+00
#>  -2.8180e-01  5.8864e-01 -5.8115e+00  1.2895e+00  1.5825e+00 -5.7390e+00
#>  -3.8966e+00 -1.0117e+01 -2.3710e+00  5.0772e+00 -2.0190e+01  7.2330e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
