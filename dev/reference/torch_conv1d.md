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
#>  Columns 1 to 6  1.7719e+00  2.4958e+00  9.0769e+00  4.8366e+00  4.4078e+00  1.5746e+00
#>  -4.7948e+00  9.5169e+00  1.1887e+01 -1.3047e+00  8.5196e+00  5.7637e+00
#>  -3.2704e+00  4.2491e+00  7.7036e+00 -2.3591e+00  8.8707e+00  5.5795e+00
#>   4.8254e+00 -4.9299e+00  6.6142e+00 -5.3441e+00 -9.8088e+00 -9.1710e+00
#>  -4.7878e+00  1.0365e+00 -1.3158e+00  6.4919e+00 -9.3174e+00  6.0574e+00
#>  -4.4428e+00 -3.5362e+00  4.8626e+00 -4.5703e+00 -1.9285e+00 -4.8209e+00
#>   4.2852e+00  1.4446e+01  4.9382e-01 -1.9303e-01  7.0296e+00  3.0479e+00
#>  -1.7659e+00  1.4743e+01 -6.1040e+00 -4.8732e-01  5.8654e+00 -3.7953e+00
#>  -7.8288e+00 -4.8002e+00  6.6289e+00 -6.6953e-01 -1.9200e+00  3.9765e+00
#>   5.0427e+00  4.2942e+00 -1.2081e+01  9.8229e+00  2.6615e+00  1.1419e+00
#>  -1.5705e+01 -2.5583e+00  5.8925e+00  2.5885e+00  2.3056e+00  2.1570e+00
#>   5.1233e+00 -3.5422e+00 -1.0569e+01 -5.3796e+00 -3.8444e-01 -7.6226e-01
#>   1.8682e+00  8.8717e+00  3.8222e+00  1.7538e+00 -1.4317e+00  4.8613e-01
#>  -9.7973e-01 -4.6300e+00 -6.4290e+00 -6.6907e-01  2.7911e-01 -2.6224e+00
#>   1.3451e+00 -1.1849e+01  2.6838e+00  2.1524e+00  4.4860e+00  6.2749e+00
#>   4.1155e+00 -1.1175e+01 -5.5178e+00 -9.0676e+00 -2.4912e+00  3.3907e+00
#>   2.1033e+00 -7.9336e-01  1.1083e+01  1.1095e-01 -6.3784e+00  4.7539e+00
#>   6.1793e+00  7.6628e+00 -1.2086e+01  8.3258e-01 -5.2913e+00 -5.6144e+00
#>  -6.7837e+00  1.1306e+00  2.6120e+00 -1.0816e+01  9.4710e-01  7.7268e-01
#>   2.3367e+00 -4.3003e-01  2.0127e+00 -1.7058e+00 -6.9291e+00 -7.5730e+00
#>  -9.7360e+00 -5.7257e+00  1.3118e+00  5.2458e-01  1.2872e+00 -9.0891e+00
#>   2.5101e+00 -4.2956e+00 -7.4874e+00 -1.0789e+01  7.0948e+00 -9.5902e+00
#>   5.1802e+00 -1.1565e+01 -2.1379e+00  2.6284e+00  3.0428e+00 -6.0112e+00
#>  -5.4779e+00  6.3713e+00 -9.4354e+00 -7.0944e+00 -3.6256e+00  3.2536e+00
#>  -8.5697e+00  1.8211e+00 -5.3610e+00 -7.0285e+00 -3.1687e+00 -9.2637e+00
#>   4.4279e+00  3.1096e-01 -7.3203e+00  5.9144e+00 -2.7811e+00  1.5250e+00
#>  -1.6007e-01  2.6723e+00  5.1532e+00 -1.3019e+01  1.7924e+00  8.4853e-01
#>   7.1994e+00 -7.3888e+00  9.8454e+00  7.7703e+00  1.3416e+00  1.7041e+00
#>   6.6906e-01  1.5630e+00 -2.5818e-02  4.5705e+00 -1.3126e-03  1.7208e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
