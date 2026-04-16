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
#> Columns 1 to 8   1.5854  -6.7376 -13.8192  -3.6195   4.2539   6.8303 -10.1406   7.9237
#>   -0.6435  -4.5192  -4.4366   6.8550  -7.1877   9.4238   6.2311  -8.3288
#>    0.0179   3.7045   2.8715  -6.9683  -1.2583  -6.2527 -19.8703   2.4047
#>   -0.9721  -3.4872  17.7279   2.4659  -5.1523  -3.3093   7.3320   6.0977
#>    5.2074   7.0991  -4.4012  -1.9502  -6.9605  28.7121   6.9774  -0.2318
#>    0.9899  14.5981  -1.2624  -8.7166  -2.1748  -8.5568  -8.7345 -25.1569
#>   -1.0761  -4.8878  -6.9622   2.9949   4.1579   7.7624  13.5768   8.8978
#>   -4.4178   2.0955  -3.3956 -14.2936  -0.8904   8.9865  -0.1400  -1.0362
#>    2.1358   2.3865 -10.0130   3.1078 -13.2532 -18.0866   4.7795  -8.5867
#>    5.0073   5.3539   3.9880   3.8234 -11.5307  25.4680   2.7772   1.4421
#>    3.2356  -9.8268  -8.6728   7.7278  -3.6339 -16.3709  20.8686  -4.8264
#>    2.6887   5.2172  -0.2614  -7.2422  -7.0155 -14.0542  -0.3986   2.4002
#>   -0.5687  -6.9324  -1.6671  -1.6553  16.5495  14.2731 -12.7583   9.1357
#>   -5.6238  -1.6098   3.6916  -8.8261   7.2654 -11.4075  -8.9379   6.0490
#>   -3.1416  12.3758   8.3161   3.7484  -3.4201   2.6698 -17.4414 -13.5820
#>    4.0855  -9.6126   3.1429   1.1349   5.3104   3.1234  -3.4529  12.5520
#>    4.7973  -8.2993   2.7627  17.5238 -27.9095 -15.3796  15.3197 -12.9976
#>    4.6027   1.1191  22.5031  -6.7222  -8.6685  -5.1421  10.2692  -4.9958
#>    4.0311  -0.4543  -5.4760  -4.3095  -6.4107  22.4609  18.3621   4.7769
#>   -1.6707   6.1465  -0.3478  -1.7548  15.9317  10.4763  -2.1930 -11.1410
#>   -3.2495  -6.3873  -7.7350  -7.6657   1.6900  23.0151   3.2388 -10.5587
#>   -0.0591  -4.3580   2.0966  -7.6219  -2.8059   0.8940  22.0035  12.0516
#>    0.2907   0.3339  -0.2539   6.9795  -2.5931  -2.3078  -8.5183  -8.7871
#>    0.0325   6.6981   1.8237  -1.8904  -5.4654  14.8174  -6.5576 -16.2769
#>   -1.3352  -0.9113   6.5780  -0.4552  13.5710   5.8022   8.6375  -6.2152
#>    1.4881  -1.8794  -6.8435  -2.0055  -0.4720  -2.1046   4.5233  -9.2600
#>    0.2594   0.5645  -1.6999  -3.2586  -7.4038  -6.9654  18.4487 -17.8512
#>    1.0255   4.9093  -9.9145  -4.2038  -8.9719   8.3787  12.0860  -6.7325
#>   -2.3152   3.6883   4.5154  -5.7578   0.3087   6.0643   7.6868 -15.1061
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
