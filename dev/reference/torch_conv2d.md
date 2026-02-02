# Conv2d

Conv2d

## Usage

``` r
torch_conv2d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias tensor of shape \\(\mbox{out\\channels})\\. Default:
  `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a tuple `(padH, padW)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv2d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

See
[`nn_conv2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
filters = torch_randn(c(8,4,3,3))
inputs = torch_randn(c(1,4,5,5))
nnf_conv2d(inputs, filters, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#>   -9.3143   4.8743   1.2288  -1.0530  -4.6163
#>   -3.3099  -9.4103  10.7395  -4.7112  -6.1624
#>    0.3354  -2.3964  11.8564  -4.3396  -0.5582
#>    1.8482   7.7441   2.4213  -2.4662  -1.4126
#>   -2.4530   1.2334  -5.2665   7.7742  -2.0204
#> 
#> (1,2,.,.) = 
#>    3.9782   4.3360  -2.1299   3.2586  -1.3188
#>   -4.6584  -4.8449  13.0824   3.4009  -4.3743
#>   -5.7788  -4.4954  -2.5234  -0.0810   1.2471
#>    5.3935  -2.0608   4.5371   6.5723   1.6264
#>    0.4323  15.2037  -3.3725   7.2405  -0.0661
#> 
#> (1,3,.,.) = 
#>    8.8386  -1.8760  -8.5144   0.9021  -0.4387
#>   -0.0960   4.4016  -3.1040  -3.5492  -8.3118
#>   -6.7808   0.9264  -8.1951  -3.6616  10.3722
#>   -5.0597  -1.7802  -1.3687   8.5094  -2.8855
#>    2.9437   6.3635   7.2899  -8.8112   1.8008
#> 
#> (1,4,.,.) = 
#>   10.2990  -3.4341   4.4002  11.0629   0.1845
#>   -1.8420   4.4368   0.0280   5.7637   2.3255
#>    3.4683  -5.5737  -5.3236   1.8088   6.4555
#>   -2.7094   3.6910   0.7525   7.6480   5.7921
#>    0.4698   7.8213  -4.7707  15.5447  -0.7094
#> 
#> (1,5,.,.) = 
#>   0.3289  0.7025 -1.3492 -6.1632 -4.3505
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
