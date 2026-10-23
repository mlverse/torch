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
#>   3.5997   1.5564  -5.8450   0.5573   3.0180
#>    5.8616   6.4098  -3.2188  -0.0196   4.1411
#>   -0.2745   7.4683  -8.5219  -3.6921   4.1657
#>   -1.9444  -0.2024  -0.8558 -11.0479   7.5999
#>    1.8795   4.1662   2.7131  -8.9813  -0.4184
#> 
#> (1,2,.,.) = 
#>   8.2799  -3.0415  -6.6606   8.3643   1.9387
#>   -8.0312  15.9917 -12.3303   9.2627  -0.5517
#>   -0.5792  -0.2185  -0.6509   7.2311  -1.0078
#>    2.2615  -3.3844   0.8231  -0.1380  -3.1337
#>    3.8663  -4.2173  -1.6354  -4.3101  -0.9914
#> 
#> (1,3,.,.) = 
#>  -8.3353  -3.8020   2.6474   6.2760  -2.1658
#>   11.9869  -8.8790   7.4101   6.3678  -1.4530
#>   -3.0123   1.6194   1.0453   9.5886  -2.1897
#>    1.9805  -0.5237   1.1887   2.3560   3.7138
#>    0.5910  -2.6551   4.2788   1.5293  -2.5586
#> 
#> (1,4,.,.) = 
#>  -5.1678  -6.0712  12.9682   2.5343  -0.1649
#>   12.6384 -10.3252   2.7374   4.1540   2.4077
#>    3.5860  -6.4822  -1.6729   1.4340   7.9282
#>    0.0643  -2.5571  -7.2524  -0.4505   6.3944
#>   -1.7054   3.6569  -8.6472   5.4093   4.8295
#> 
#> (1,5,.,.) = 
#>   1.5255 -12.9960   4.4716   0.4027  -9.0158
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
