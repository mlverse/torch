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
#> Columns 1 to 6  1.1455e-03 -3.3740e+00  4.6615e+00 -2.0050e+01 -4.4363e+00  6.5586e+00
#>   5.4455e-01 -3.7714e-01  1.5816e+00  4.3998e+00 -4.7111e+00 -4.4354e+00
#>  -1.8509e+00  1.8192e+00  6.1506e+00  4.1651e+00 -3.3145e+00  4.2449e+00
#>   7.8741e-01  2.2341e+00  1.5689e-01  4.2962e+00 -1.0624e+01 -1.6866e+01
#>   5.1305e+00  1.3146e+01 -1.0559e+01 -1.4600e+01 -9.8598e+00 -1.2936e+01
#>   3.5332e+00 -1.1508e+01  1.0635e+01 -1.4468e+01  1.1651e+01 -3.1873e+00
#>   1.1917e+00 -1.6113e+01  9.3093e+00  8.1815e+00 -1.4063e+01  3.1534e+00
#>   5.1224e-01 -1.1866e+00 -2.6177e+00  3.4269e+00  3.3335e+00 -5.1976e+00
#>  -5.0162e+00 -3.9612e+00  7.0659e+00 -4.9123e+00 -4.7546e+00 -1.0915e+01
#>  -4.0060e-01  2.4546e+00  5.6006e+00  5.1523e+00  1.4688e+00  1.3014e+00
#>   1.1807e+01 -1.2504e+01  9.5956e+00 -2.8686e+00 -1.6228e-01 -1.2168e+01
#>   1.9168e+00 -9.6683e+00  1.4650e+00 -7.6029e+00  3.7807e+00  2.6370e+00
#>  -2.6051e-01  2.3013e+00  2.7390e+00 -6.3337e+00 -1.6113e+01  8.4174e+00
#>  -4.5547e-01 -4.9066e+00  5.4635e+00  1.6996e+00 -1.4836e+01 -4.6891e+00
#>  -1.0648e+00  1.6421e+00  4.5397e+00  6.1223e+00 -1.0616e+01 -6.2939e+00
#>   1.3490e+00 -1.6026e+00 -9.6883e+00  8.2475e+00 -1.1182e+01  8.0345e+00
#>   5.7866e+00  6.1125e+00  5.6325e+00 -5.2155e+00  5.3727e-01 -4.6130e+00
#>  -3.5815e+00  7.2677e+00 -6.8870e+00  3.6288e+00  4.2572e+00  5.4431e+00
#>   2.5129e+00  1.3228e+00 -4.1008e+00  6.9314e+00  4.8305e+00 -1.8312e+01
#>  -1.7207e+00  5.5748e+00  3.3993e+00  2.0160e+00  5.4101e+00 -7.5340e+00
#>  -1.0009e+00 -4.1552e+00  1.5045e+00 -5.7818e+00  7.0984e+00  1.4613e+00
#>  -5.7892e+00  4.3758e+00  2.8504e-01 -8.6829e-01  1.0299e+01  4.4144e+00
#>   3.8199e+00  6.5786e+00 -9.7157e+00 -5.0030e+00  4.5604e-01 -8.9866e+00
#>   1.4919e+00 -4.8823e+00  3.7515e+00  3.4307e+00 -6.7423e+00 -1.3844e+01
#>   8.1807e-01 -1.3885e+01  2.3051e+01  7.6020e+00 -7.1292e-01 -1.6063e+01
#>  -1.6234e+00 -1.4551e+00 -5.9453e+00  4.2033e+00  6.6013e-01 -8.7411e-01
#>   5.9553e+00  7.3720e+00 -2.9123e+00 -1.2934e+01 -1.4603e-01  4.2590e+00
#>   3.5002e+00  3.2573e+00  6.2935e+00  1.3773e+01 -1.4211e+01 -6.9339e+00
#>  -1.3866e+00 -2.9453e+00 -1.7027e+01  1.3872e+01 -3.0626e+00  1.8781e+01
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
