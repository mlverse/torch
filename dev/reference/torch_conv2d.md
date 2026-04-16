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
#>   0.9089 -10.8567  -1.9894  -0.9382 -13.4880
#>    0.0734   1.1064 -14.5551  -8.6304  -2.3498
#>   -5.5097   1.5078   3.2027   5.3682   5.5200
#>   -1.7565   4.8423   2.4430 -14.9831  -2.4674
#>   -8.6273  -3.0938  -8.2263   5.7800  -0.8637
#> 
#> (1,2,.,.) = 
#>  -1.8555  -1.7106  -0.2407  -6.0109  -1.6191
#>    4.6923  -7.0027  -3.9986  -1.0112  -3.3595
#>    3.6166  -2.9814  -2.0398   7.0241  -7.1219
#>    4.5189   2.0097   2.7329 -10.5421   8.1524
#>   -2.5508  -2.6574  -3.9852  -4.1742   0.1321
#> 
#> (1,3,.,.) = 
#>  5.2056 -3.9391 -0.7699 -4.2931 -3.2150
#>  -4.7735 -2.9935  0.2571 -1.6739  9.8436
#>  -2.2106 -9.6423 -9.8559 -3.8997 -1.9083
#>  -3.9342  4.8431  2.8939 -3.2811  2.5383
#>   2.5974  6.9373 -2.3465 -4.0793  6.2187
#> 
#> (1,4,.,.) = 
#>  -6.0511   0.0650  -5.5357  -3.9945   0.8852
#>    0.3294  -1.0962  -1.6240 -10.7574  -1.3803
#>    2.1602   7.6392   2.2909   8.1189   2.7716
#>    0.6947  -2.9175  -2.4012  -5.1725   0.5278
#>    1.1331  -0.9321   0.5739  -4.6575  -5.2235
#> 
#> (1,5,.,.) = 
#>  -0.0201   6.6629   5.0407  -5.7274  -0.8883
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
