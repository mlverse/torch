# Conv_transpose2d

Conv_transpose2d

## Usage

``` r
torch_conv_transpose2d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} , iH ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kH , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sH, sW)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padH, padW)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padH, out_padW)`.
  Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dH, dW)`. Default: 1

## conv_transpose2d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose2d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose2d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
inputs = torch_randn(c(1, 4, 5, 5))
weights = torch_randn(c(4, 8, 3, 3))
nnf_conv_transpose2d(inputs, weights, padding=1)
}
#> torch_tensor
#> (1,1,.,.) = 
#> -0.6887  1.9960 -0.1364 -1.2844  0.4219
#>  -1.4909  5.8200  2.6576 -5.0277  2.9035
#>  -0.1233 -1.0895  3.1966  5.4611  3.5587
#>   0.2288  0.5348 -0.9942  2.3831 -3.6646
#>   1.6663 -2.4611 -3.6768  2.3582  1.9017
#> 
#> (1,2,.,.) = 
#>  -0.0415  -0.7797  -4.4809  -3.1085   2.2628
#>   -5.2519   9.9390  -9.3326   1.6631   0.2891
#>   -0.1864  -8.9767  -4.6772  -2.1519  -0.0761
#>   -1.5821  -6.6150   6.6121   1.1982  -1.7620
#>  -11.8406  -0.6828   5.9532   1.6505  -1.8327
#> 
#> (1,3,.,.) = 
#>  -5.3813   0.8080   2.4750   9.8796   3.0824
#>    2.4882   5.4502   8.5764  10.8994  -7.8376
#>   -1.2071   5.8477  11.3334  -0.3233   0.2523
#>    0.1763   3.1001   4.5515  -3.7741   6.3074
#>   -5.0046   5.8014   0.0081  -2.6620   2.6035
#> 
#> (1,4,.,.) = 
#> -4.7803 -4.2553 -4.5166  7.3688 -1.2693
#>   4.3285 -3.7721  6.7917 -9.2343 -5.1372
#>   0.0929 -8.7484 -2.4925 -5.2435 -2.0901
#>  -4.0879  9.7927 -2.5732 -2.0738 -6.2609
#>  -9.5730  4.5497 -0.0094  4.8636  2.6965
#> 
#> (1,5,.,.) = 
#>  1.5848 -2.9264  1.2675  2.5381 -4.3057
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
