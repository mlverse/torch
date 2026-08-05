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
#> Columns 1 to 6 -3.6337e+00 -5.7185e+00 -3.2913e+00 -1.7157e+00 -9.7542e+00 -5.1414e+00
#>   3.2902e+00  1.8404e-01 -4.9955e+00  1.7572e+00  4.3920e+00 -9.0420e+00
#>   2.8378e+00  2.3733e+00 -5.1009e-01  6.4420e+00  2.2003e-01  4.5824e+00
#>  -3.7464e+00 -2.7069e-02  4.0236e+00  1.7031e+01  1.4990e+01  1.6475e+00
#>  -4.6378e+00  6.6693e+00 -2.8617e+00 -2.6492e-01 -1.0627e+00 -8.2345e+00
#>  -3.7596e+00 -4.6023e-01  9.1557e+00 -2.8856e+00  1.6195e-02  3.4992e+00
#>  -1.7996e+01 -1.4016e+01 -6.9592e+00 -1.2287e+01  9.5689e+00  4.0038e+00
#>  -6.6773e-01  9.2842e+00 -6.8665e+00 -3.6664e+00 -3.4067e-01  2.2626e+00
#>  -9.9812e+00  7.1085e+00  4.9418e+00  1.2019e+01  2.6686e+00  9.9515e-01
#>  -3.7268e+00  4.2636e+00  1.1362e+00 -9.8557e+00 -2.1850e+00 -3.0994e+00
#>   1.0610e+01  1.1769e+01 -2.5880e+00 -2.0888e+00 -3.6925e+00 -6.2875e+00
#>   4.6707e+00 -1.0917e+01  4.4717e+00 -4.2100e+00 -5.2351e+00  8.7873e+00
#>   7.2677e-01 -6.1983e+00 -6.3472e+00  6.3538e-01  7.9071e+00 -4.4365e+00
#>   5.3297e+00 -9.5827e+00 -6.0956e+00  4.3432e+00  6.0625e+00  7.2530e+00
#>   1.6168e+01 -1.1671e+01 -8.5591e+00 -6.4357e+00 -5.1592e+00 -1.4003e+00
#>  -9.1819e+00  1.5588e+01  9.0723e+00  1.3135e+00  1.5127e+01 -4.1238e+00
#>  -4.1785e+00 -1.5446e+01  9.2562e+00 -9.4678e+00 -2.4989e+00  5.5722e+00
#>  -1.5151e+00  2.9549e+00  3.7320e-01  9.7091e-01  3.8029e+00  1.9179e+00
#>  -1.8133e+00 -1.7304e+00 -7.1675e+00 -3.2021e+00 -7.6534e+00 -8.1997e-01
#>  -4.6565e+00 -3.3874e-01 -9.6424e+00  1.1530e+00  3.5222e+00  1.1809e+00
#>   1.0486e+00 -1.4922e+00 -3.8255e+00  2.0637e+00 -1.2870e+01 -4.6797e+00
#>   2.4202e+00 -6.0655e+00  1.6088e+00 -1.4671e+00  6.2483e+00  8.6022e+00
#>   1.4970e+00  5.6540e+00  1.1694e+01 -5.4359e+00 -6.5628e+00  5.7447e-01
#>  -1.9809e+00  2.6466e+00  7.1522e+00 -2.1556e+00  3.1115e+00  5.5109e+00
#>   3.7152e+00  4.5717e+00  1.5797e+00  6.4896e+00  4.4855e+00  3.2433e+00
#>   8.8470e-01 -1.2972e+00 -6.1133e+00 -3.5562e+00 -3.9815e-01  1.6041e+00
#>  -3.9523e-01 -1.3865e+01  4.5964e+00 -5.8897e+00  2.7651e+00  3.0639e+00
#>   4.1650e+00  4.5415e+00  4.0545e+00  3.2221e+00 -1.1669e+01 -7.8134e+00
#>   7.6019e+00 -5.0099e+00  1.1707e+01 -5.3830e+00 -2.5953e+00  1.0701e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
