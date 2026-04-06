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
#>  -2.1820   2.3510   0.1111  -5.9752   1.9525
#>    4.7720   2.1245  -2.3314  -2.1174   5.6945
#>    2.5693  -1.1678 -14.6034   0.4942  -2.4574
#>    3.2854  -3.4527  -1.9588  -1.6726  -2.7589
#>   -2.3144   7.6551  -1.2029 -14.7026  -1.9140
#> 
#> (1,2,.,.) = 
#>   3.2883   4.8650   0.7833   5.0532   1.5175
#>   -2.3669 -10.8024   1.5425   0.9174   7.7926
#>    3.4221  -0.5809   8.6117 -21.3776   2.9279
#>    0.1832  -0.5653  -8.0658  -7.1090   1.4010
#>    1.5375   4.1447  14.5674   0.1437  -3.2568
#> 
#> (1,3,.,.) = 
#>   0.4897  -2.5692  -5.9019   3.2515   1.8690
#>   -2.6622  -2.4085  -0.6893  -1.3538  -6.4777
#>    1.4870   3.9685  -0.6545  -5.5082   0.0109
#>    1.3864   5.1712  -1.1732  11.7975   4.3794
#>   -2.2954   6.1755   0.8802  -2.0079  -0.2110
#> 
#> (1,4,.,.) = 
#>   9.3201  10.8931   7.8444  -1.5496   2.7357
#>    1.6921   1.9738  -1.5051   9.2737   8.0769
#>   -1.5900  -3.7577  -4.3505   2.6733  -5.9897
#>    0.6569  -4.4721   0.4953 -10.1937   1.1127
#>    0.3195  -2.9783  -1.4194  -3.4810   5.2815
#> 
#> (1,5,.,.) = 
#>  -6.0897 -11.1819  -5.3519   8.1727  -4.5138
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
