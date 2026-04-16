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
#>  -6.9157   2.0553  -0.2081   2.4811  -2.9416
#>   -2.6744  -0.8432   1.5180  -2.1141   4.2651
#>   -1.3429   4.3386  -1.7667  -0.1960  -2.3261
#>    3.0600   5.4606   7.6038  10.5917  11.0434
#>    3.3583   0.8759   3.8877   2.7023   5.2057
#> 
#> (1,2,.,.) = 
#>   1.6952  -5.8988   0.4306   3.8386   0.2730
#>   -2.1049  -0.8813   6.0531 -12.9343   8.6866
#>   -4.3120   2.7993  -0.1080  -7.8279  -4.6148
#>    5.4190   5.9546  -8.8110   9.3736   2.3182
#>    6.6866  -2.5727   1.3452   0.8510  -2.1986
#> 
#> (1,3,.,.) = 
#>   1.6793  -1.0251  -3.0154  -2.3419  -0.2083
#>   -5.2576   3.8983   1.5911   1.3740  -5.2096
#>   -3.3643   0.9740  -0.3285   5.5017  -0.3279
#>    0.3013  -1.5352  -6.9418   4.0598  -2.3059
#>    4.4546  -2.6464   6.5992  13.2742   4.0030
#> 
#> (1,4,.,.) = 
#> -1.7067 -7.3991  2.0961 -4.9155 -1.5148
#>   4.4223  2.2897  2.3202  4.0903  6.1425
#>   2.8113 -4.1337 -9.0247 -6.3580 -4.3121
#>   3.2704 -1.6323  0.8848  5.1174 -3.7715
#>  -4.8334  0.9748  4.0616 -7.9000 -5.4348
#> 
#> (1,5,.,.) = 
#>   4.5329  -1.4137  -9.6794   0.7527   5.0254
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
