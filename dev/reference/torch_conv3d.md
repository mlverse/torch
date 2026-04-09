# Conv3d

Conv3d

## Usage

``` r
torch_conv3d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iT ,
  iH , iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kT , kH , kW)\\

- bias:

  optional bias tensor of shape \\(\mbox{out\\channels})\\. Default:
  NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sT, sH, sW)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a tuple `(padT, padH, padW)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dT, dH, dW)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv3d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 3D convolution over an input image composed of several input
planes.

See
[`nn_conv3d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv3d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# filters = torch_randn(c(33, 16, 3, 3, 3))
# inputs = torch_randn(c(20, 16, 50, 10, 20))
# nnf_conv3d(inputs, filters)
}
#> NULL
```
