# Conv_transpose1d

Conv_transpose1d

## Usage

``` r
torch_conv_transpose1d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sW,)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padW,)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padW)`. Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dW,)`. Default: 1

## conv_transpose1d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

inputs = torch_randn(c(20, 16, 50))
weights = torch_randn(c(16, 33, 5))
nnf_conv_transpose1d(inputs, weights)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 8   3.4566  -2.0889  -5.9552  -8.5269  -5.8477   9.7308 -11.1417  -4.3085
#>   -1.2214  10.3842   3.9630 -12.0067 -17.6492  -3.2668  -4.4285  -1.0120
#>   -0.8319   0.9440   5.1075   9.9345   4.9070   9.7983 -12.3350   6.1523
#>   -5.3152   1.8917  -3.4846  -2.6862  10.8680  -4.8150  -6.3534   9.9539
#>    1.1347   1.5308  -0.4626  -9.6633   0.5746   0.9625   3.2856   2.5286
#>    1.6191   3.2719  -1.5949   3.9286  -2.5958  -8.5780  15.0900   3.8745
#>    3.0238  -6.0896   5.4877 -12.4152  -1.0278   6.0534   3.6389   3.7165
#>    2.9461  -7.2199   6.0783  -4.4634  -9.5637 -17.7887  -3.7480  -9.6704
#>   -1.0973  -7.7957  -3.5418   2.2955   4.0530   4.8437 -17.3016  -4.3677
#>    1.9071   7.8795  14.3990  -4.9124   1.0720  11.4232  11.2508  -4.8563
#>   -2.5586  11.0108   6.1903 -11.7860   4.7121   0.0917  -1.3785   4.9678
#>    0.0237   9.0203  -4.7017 -12.0089   5.2802 -21.8514   1.0757  -8.2715
#>    1.1986   4.2149  -1.7505 -16.6591  -1.9255  -3.0808 -10.7640  -0.1494
#>   -2.4708   1.0923   1.4334  -1.2436  -4.2132   2.3486  11.4337   0.9906
#>   -1.6261   3.1320   4.5954   2.1644  21.7644  -7.1863  -5.3868  -7.1330
#>   -6.8603   6.8454   0.2131  -0.6991  17.9365 -16.3094   7.9413   2.6827
#>   -1.6339  -0.1886  -8.6059  -4.9470   3.8219 -13.4624   2.2472  -4.8966
#>    4.6876  -8.5541  -6.9441   6.8635 -10.6867   4.5328  -7.3114 -11.7141
#>   -7.5499   1.8576  15.4309  -2.3331   0.8286   0.6428  12.2199  23.4378
#>    1.5169   1.4778  -5.7002   5.2739  21.4425 -17.1571   1.5128  -3.8076
#>   -3.8688  -2.3043   5.7136  14.9039   6.0190  22.0930  -3.0293   3.9443
#>   -0.3535   5.5065   9.1429   1.7262  -3.0508   9.2089   2.1884   3.4022
#>   -3.9227   9.3231  -9.1199 -18.6622   7.7849  -0.6758   1.6155  -8.8215
#>   -1.9572  -1.1679  -3.3051 -10.7269  15.4518  -4.5318 -18.2861   5.7472
#>    2.8288   5.5552   3.4568  -1.3359  -7.6750   6.2739   7.4029  -9.3993
#>   -0.4620  10.6362   4.5463  -2.8330   7.1343  10.6977   2.3665  -1.1106
#>    0.8305   1.0393 -20.6133  12.9695   3.0069  -7.2299  -3.4798  -1.8182
#>   -0.2140  -2.3831  -2.6656  -1.6116  -4.1954   0.1863  -5.4729  10.6291
#>    1.8344  -3.3594   9.3134   3.8933  -6.2659   1.2718  16.7379   8.2982
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
