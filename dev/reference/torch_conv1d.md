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
#> Columns 1 to 6 -2.4448e+00 -3.7506e+00 -6.0645e+00  7.5288e+00 -5.2332e+00 -2.8787e+00
#>  -3.1346e+00 -9.6420e+00 -7.3853e+00 -3.7098e+00  2.3960e+00  2.7032e+00
#>  -8.8477e+00 -8.8116e+00 -1.1354e+01 -1.5734e+01 -5.1093e-01  1.2039e+00
#>  -2.5263e+00 -6.1661e+00 -1.3639e+01  8.0133e+00  3.9859e+00  4.6400e-01
#>   5.1907e+00 -1.1263e+01 -1.8133e+00  1.0809e+01 -4.6561e-02 -8.0000e-01
#>   5.8535e+00  1.1109e+00 -6.8222e+00 -4.4122e+00 -1.2138e+01 -1.4266e+01
#>   2.7600e+00  2.3104e+00 -6.8157e+00 -1.2393e+01 -5.4912e+00  4.7464e+00
#>  -1.0363e+01  8.4431e-01  1.4193e+01 -8.5693e+00  9.1858e-01 -4.3452e-01
#>  -8.3154e+00  8.4461e+00  1.5851e+01 -1.2801e+00 -8.1898e+00  1.8008e+00
#>  -4.6904e+00 -6.6396e+00  6.4231e+00  2.5877e+00 -4.3625e-01 -8.0274e+00
#>  -3.4031e-01 -1.5359e+01  7.9515e-01 -6.4940e+00  1.0963e+00  9.0244e+00
#>   5.3391e+00  1.5053e+00  6.2064e+00 -9.6148e+00 -6.0499e+00 -4.8660e-01
#>  -6.8086e+00 -1.1496e+01  1.2515e+00 -7.0637e+00 -1.1374e+00  4.1131e+00
#>  -5.4107e-01 -2.6899e+00  1.2230e+00  3.7985e-02 -6.3644e-02 -1.7045e+00
#>   1.1549e+00 -7.0013e+00  1.1275e+01 -8.5472e+00 -4.5850e+00 -2.3960e+00
#>  -8.3000e+00  1.3840e+01  2.6801e+00  6.3858e+00  2.9384e-01 -3.5596e-01
#>   2.7026e+00 -8.4556e-01 -3.3343e+00 -1.0912e+01 -9.2909e+00 -4.3421e-01
#>   9.9024e-01 -7.9836e+00  3.5941e+00 -5.9234e+00 -4.3788e-01 -6.9700e+00
#>  -1.2562e+00 -3.7811e+00  1.5640e+00 -1.4225e+01 -7.4200e-01 -1.6710e+00
#>  -5.3419e+00  8.3787e-01  3.3918e+00 -8.8676e+00  5.3372e+00  2.8585e+00
#>  -3.0212e+00 -6.5670e+00  1.1716e+00 -7.7954e+00 -5.7899e+00 -6.6519e+00
#>   4.3911e+00  8.3903e+00  3.5403e-01 -1.1283e+00 -4.7320e+00 -8.3236e+00
#>  -8.5784e-01  2.9597e+00 -1.6606e+00 -1.4938e+01  1.5459e+00  4.7654e+00
#>  -1.0844e+01  4.9796e+00 -1.5931e+00 -5.9996e+00 -7.3563e+00 -4.9477e+00
#>   8.7104e+00  1.3597e+01 -3.9249e+00 -3.5028e+00  4.4130e+00 -1.9440e+00
#>   3.0350e+00  4.8248e+00 -1.1553e+00  3.8272e+00 -6.6106e-01 -2.7614e+00
#>   2.6154e+00 -7.2777e-01  3.0768e+00 -1.5639e+01 -1.2099e+01 -3.0934e+00
#>  -8.0260e+00  3.6978e+00 -3.7895e+00 -6.9376e+00 -6.0707e+00 -9.7141e+00
#>   8.5611e+00 -3.5562e+00  6.1540e+00 -2.5326e+00 -1.6753e+01 -4.8287e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
