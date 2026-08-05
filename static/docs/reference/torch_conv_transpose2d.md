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
[`nn_conv_transpose2d()`](https://torch.mlverse.org/docs/reference/nn_conv_transpose2d.md)
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
#>  -4.3016  -0.8949   3.5076   0.4936   2.9163
#>   -7.0659  -5.4800   1.8480   2.0342  -0.3964
#>   -3.2120   2.2482  -3.5073   1.2345  -2.3635
#>   -8.0329  10.5966  -3.1365  -5.6299  -9.5521
#>   -1.3244   1.4126  -5.8069  -2.6295  -6.4017
#> 
#> (1,2,.,.) = 
#>   1.4985  -3.7923  -7.8212  -6.4652  -2.4251
#>    2.0866   6.2682   6.4722  12.2570   4.6922
#>   -5.3317  -1.9319  -1.3015  -5.2165  -4.4574
#>   13.5155  -3.9603  -3.5371  -1.7429   2.1187
#>  -12.2307  -8.3822   1.6126  12.3699   6.5261
#> 
#> (1,3,.,.) = 
#>  2.7311  0.1965  2.8391  4.1715 -3.8729
#>  -6.7028 -9.9584  0.5762  9.7599  6.1506
#>   3.5282  1.2264 -5.5813  0.5293  3.3439
#>  -1.8130 -2.2540 -5.4678 -9.6381  2.7170
#>   0.3313 -1.8392 -2.0380  1.7353  0.5872
#> 
#> (1,4,.,.) = 
#>  -1.8508  -3.5694   3.4386  -4.3745  -2.7689
#>    2.7275  -5.2901   1.5518 -10.9613  -0.7322
#>   -1.9250   1.8529  -6.6072  -1.2228  -2.9147
#>    3.8800   0.8301   8.9261   9.9140  -1.2501
#>    0.6717   3.1771   3.4518  -0.2347   1.0939
#> 
#> (1,5,.,.) = 
#> -3.8390  3.1929  2.0380  6.4444  2.0707
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
