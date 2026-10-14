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
#>  -0.1352  -5.2671  -5.3552   2.0686   6.0390
#>   -1.5140   0.6928   0.2918  -3.5425 -10.7799
#>   -1.9756  -3.7219  -6.3375   6.8803   3.2578
#>    5.5009  -2.9181  -8.4664   2.5298   7.9981
#>   -0.9992  -4.0875   5.9830   6.1162  -0.5037
#> 
#> (1,2,.,.) = 
#>  3.2463 -4.7487 -5.3296  5.6722  5.4716
#>  -0.6035  6.8022 -5.1244 -5.2049  1.0470
#>  -1.3793 -2.4754 -2.4722 -2.0036  3.6554
#>   1.5493  2.5520 -5.6285  1.4964 -4.0000
#>  -1.6750  5.8555  0.1572  4.6426 -5.8215
#> 
#> (1,3,.,.) = 
#>   2.2695   5.3716  -1.7159  -4.8684  -3.4815
#>    0.0301   0.7494   0.9806   1.1203  10.3669
#>   -3.6244  -4.3380  -4.3189  -3.3760  -3.4660
#>   -3.5020   2.0796   0.6676  -0.0663  -8.8517
#>    4.9362  -7.1031   0.8695  -3.7571   5.6980
#> 
#> (1,4,.,.) = 
#>   4.0958  -7.7126  -8.8599  -6.1893   1.9444
#>   -4.7436  -6.3022   0.4488   5.1499   0.8722
#>   -3.6046  -4.0838  -5.8929   7.1926   4.9894
#>   16.3951   2.8640  -5.1809   0.7112   3.7588
#>    2.1425  -3.5336   5.6638  12.2716   6.0751
#> 
#> (1,5,.,.) = 
#>  -0.7843   3.6169   1.3423  -8.7705   7.4318
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
