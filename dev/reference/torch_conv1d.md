# Conv1d

Conv1d

## Usage

``` r
torch_conv1d(
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

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a
  one-element tuple `(sW,)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a one-element tuple `(padW,)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv1d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See
[`nn_conv1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

filters = torch_randn(c(33, 16, 3))
inputs = torch_randn(c(20, 16, 50))
nnf_conv1d(inputs, filters)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 8   3.5797   5.2040  -3.1354   3.7161  11.6625  -8.1999  -1.1146   2.1837
#>   12.5539  -8.9367   9.5462  10.5316  -2.5722 -12.0782   0.7640   1.8701
#>    7.1642 -14.4840   6.7473  -9.8065  -0.8935  -0.0441   2.5113   1.1245
#>   -2.5796   1.3052  -4.0747  -0.4093  -3.8182  -0.7081  -0.8263  -3.7738
#>    5.5179  -1.2161  -4.7194   9.1825   4.4233  -0.0501   5.7523  -4.3860
#>   -0.0350 -11.0170  -0.7514 -18.9434   5.5668   5.4057   2.2162 -13.7331
#>    7.8453   2.4357  -1.5569   4.0637  -1.8000   6.1770   2.7547   6.6565
#>   12.3126  -7.0053  -2.2374 -11.3688   2.4835  -0.2353  -3.5038   4.4602
#>    6.6175   6.4089  -0.8132  -7.9811 -12.5216  -3.7162  -1.6546  -4.1486
#>    3.4028  -3.4386  -5.7239   5.5058 -11.6023  10.6007   3.4589  12.0945
#>  -13.8902  11.3486  -1.2518  12.0836   2.1157 -13.2719  -3.0464   2.8410
#>    0.8874   2.5316  -0.2576  -0.2329  -9.5039  -4.9829   6.4004  11.3489
#>   -2.6432  -6.4383   2.2512   0.6056   8.7229   6.6258  -4.4936  -5.7916
#>   -1.7725  -3.9233  13.7793  -1.0552  -6.8489  -4.7896  -0.8366  -0.6084
#>    1.3400   7.3511 -10.5939  11.7576  -1.6745   7.9408  -3.1778  -0.5485
#>    2.5607  10.8318   2.6897  -4.2740   8.6773 -14.4429   1.4000   3.0629
#>    1.4724   6.9016   3.7495  -1.9354   3.3252 -13.6698   3.7612   5.6380
#>  -11.4748   6.2937  -6.0607  -9.8616  -7.2656   6.4254  -1.4106   5.4971
#>   11.3970   6.4305   1.4090  -4.2922  -0.9754   4.1179  10.1855  -2.8829
#>    5.0043  -3.3701  -5.1876   2.1132   6.3154   1.3739 -16.7583  -9.3731
#>   -4.2038  20.7988  -3.2707   4.7376  -1.4020  -0.4748  -2.3698   4.8599
#>    1.4086   5.0694  -3.4204 -15.8492   6.0184  -1.7786   1.2516   2.2788
#>   -5.8762  10.1016  -7.8521   3.4813  -3.9495   3.7521  -2.1294   3.3331
#>    0.1097   1.4280  -3.9672   2.0527  -6.5183   9.0379  -6.9245  -8.5684
#>    0.2646 -13.5005  -2.6768   2.0353  -7.1655   3.2924  -5.6237  -4.8406
#>    9.3397   5.7545  -8.1281   2.0665  -1.4202  -6.0227   1.0421   3.9384
#>   -9.5267  -4.1681  -6.4915   2.5359 -13.4164  -2.3356  -0.0039   3.7242
#>    0.8092   5.2509   2.7983   4.9329  12.0041  -0.2659  -1.3089  -1.9232
#>    4.1166  -6.3171   8.6896   4.7028  -0.9879  -6.2635   5.4878   0.3081
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
