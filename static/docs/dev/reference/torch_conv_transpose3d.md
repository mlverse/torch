# Conv_transpose3d

Conv_transpose3d

## Usage

``` r
torch_conv_transpose3d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iT ,
  iH , iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kT , kH , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sT, sH, sW)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padT, padH, padW)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple
  `(out_padT, out_padH, out_padW)`. Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dT, dH, dW)`. Default: 1

## conv_transpose3d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 3D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution"

See
[`nn_conv_transpose3d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose3d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
inputs = torch_randn(c(20, 16, 50, 10, 20))
weights = torch_randn(c(16, 33, 3, 3, 3))
nnf_conv_transpose3d(inputs, weights)
} # }
}
```
