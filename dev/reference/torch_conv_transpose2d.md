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
#>   0.6626  -0.8204   4.8724   0.6926  -4.8875
#>    1.5300  -6.3139  -2.4666  -3.0087  -8.4471
#>    4.5007  -0.0875  11.8125  -5.0891  10.2696
#>   -2.0439  -4.2059   4.6838  -6.0307   1.1074
#>    0.2525   9.3690   6.2515   4.7695  -1.1814
#> 
#> (1,2,.,.) = 
#>  -0.1132  -4.1667 -10.4181  -6.3044   2.1002
#>    1.2896   2.5389  -7.4143  19.0590  -3.7667
#>   -0.7907  -3.7121   3.8816   6.8851   7.0501
#>    5.8878  -5.9189   6.2949  -7.7517  -1.4976
#>   -5.5920   0.3163   2.8995  -0.7558  -4.6996
#> 
#> (1,3,.,.) = 
#>  -1.0385   3.3834   1.7618   2.3337 -15.4483
#>    1.4075   0.3130   1.2170  -1.4921   5.9343
#>    7.5704  -5.4442  -1.0393  -4.1510  -3.5888
#>   -2.7905   6.4909  10.0944   0.4754  -0.1838
#>   -2.7906   8.0948  -0.8571  -1.5299   0.4043
#> 
#> (1,4,.,.) = 
#>  -1.5929   1.3727  -0.5876   7.7898   2.6035
#>    2.7016  11.8285   3.0719  -0.4009   4.6925
#>   -6.1753  -6.4898   5.8397   2.9588  -3.5152
#>    0.6186   1.2767  -3.9643   5.4087  -1.9654
#>    3.3304  -7.6763  -8.9729   5.2629   0.2851
#> 
#> (1,5,.,.) = 
#>  -1.5475  -1.0578  -2.2065   3.4455   1.5039
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1,8,5,5} ]
```
