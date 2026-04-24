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
#>  -2.3837  -2.8522   5.1599  -5.5307   3.7854
#>    2.7797   3.7083  11.7488  -5.8604  -2.7729
#>   -0.4998   0.2368   3.6308  -8.5296   4.6445
#>   10.4182   3.3924   4.8121  -3.2161   1.0448
#>   -4.5452 -13.4823  -6.1708  -3.0597   1.1214
#> 
#> (1,2,.,.) = 
#>  -7.6784   0.9253  -0.7998  -0.2102   2.6055
#>   13.8624  -4.1745  -2.8465   0.9078  -6.1523
#>   -6.4273   0.8340  -4.2976   1.6884   1.8859
#>    2.4354  -6.3233  -2.1190   6.3282  -4.5520
#>   -1.4674  -0.0614   1.6330   0.1505   1.5927
#> 
#> (1,3,.,.) = 
#>   6.9992  -0.4569  -3.7607  -5.0701  -3.8486
#>    7.1226 -14.4307  -0.9893   4.6448   1.2988
#>    8.4960  -7.5786   1.2449   4.5275   2.8035
#>    1.5686 -16.6238   3.8926  -2.3988   1.2738
#>    7.8427   4.7472   6.1983   6.3564  -3.5960
#> 
#> (1,4,.,.) = 
#> -4.9990 -1.6007  3.5987  8.6630 -0.2626
#>  -4.2679 -7.7756  1.2733  0.0005 -1.7259
#>   4.0010 -1.7564  9.7425  5.1837  2.2130
#>  -7.3218  5.5154 -2.9714 -1.7613 -1.3806
#>  -6.4323 -2.5802 -0.1620 -4.9876 -3.3100
#> 
#> (1,5,.,.) = 
#>   3.2502  -7.5255  -0.0154   2.0137  -2.3208
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
