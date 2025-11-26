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
#>   -1.2954 -11.1186  -9.3198   9.4092   1.1764
#>    2.0363  -0.8181  -3.7218  -3.8377   3.6143
#>   -2.5002  -5.3939  -5.9626  -2.4684  -4.7641
#>   -1.0273  10.2069   1.9019   4.0388   3.8612
#>    0.1680  -3.4011   2.9773  -7.5833  -0.4359
#> 
#> (1,2,.,.) = 
#>  -5.7749  9.5826 -4.2967 -6.1795  3.9991
#>   4.9513 -3.5873 -1.8598  6.0553 -0.6615
#>   0.6306  7.5649 -4.5200  0.8332  6.4504
#>  -3.0518 -12.3110  4.2792  0.7566 -1.2140
#>   3.2247 -7.3775  4.3529  1.0658  4.8394
#> 
#> (1,3,.,.) = 
#>   2.4743  0.5507 -4.9213 -5.4319 -0.0074
#>  -3.9592 -7.8668 -1.5536  0.4640  2.0384
#>   6.6813 -1.3527  8.9206 -0.4431  3.0639
#>  -5.1062 -7.6328 -1.4273 -0.4379  0.1977
#>   0.7581  2.2223  3.0930  5.9943  2.7056
#> 
#> (1,4,.,.) = 
#>  -1.8196 -4.3846 -1.7217 -2.3869  1.7912
#>   5.5400  8.1682 -2.6666  0.4820 -1.3295
#>  -2.0737 -0.2393 -3.8612 -8.0865 -1.7715
#>  -1.8213  0.5427 -4.0238  4.0633  4.9098
#>  -4.1490  1.9802 -1.9898  2.5311  0.3419
#> 
#> (1,5,.,.) = 
#>   2.5451 -4.7841 -1.7718 -3.4094 -2.2389
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
