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
#> Columns 1 to 8  -5.8800  -4.2851  -1.1756   8.4408   4.2652 -11.0206  -1.0238   0.1166
#>   -4.4392   3.9072   0.8895   2.2437   4.6294  -5.1261  -8.5509   5.0558
#>    3.6876   1.0376   1.0174  -3.8653  -3.9802  -6.5847  -0.8148 -13.4945
#>  -15.0410   4.1720  -2.1471  -3.2372   4.9142  -9.5616  -7.5638  -9.0786
#>    5.3457  -0.2275  -2.7376  -0.6204   3.1455  -5.2411   0.6917   9.1045
#>    2.1158  -0.6460 -10.9718  -8.4972   5.3623   3.4050   5.8550  -2.3437
#>    7.1025   0.1156   5.2376   0.1136   6.6129   2.3471  16.5063   7.1388
#>    5.9229   5.3782  -1.5082   5.4056   0.6868  -8.2204  -8.5621  -2.9187
#>   -6.4228   9.8367   6.3828   0.0900  -0.5256   7.3660  -5.2047  -2.8790
#>   -0.1383  -6.4377  10.4121   0.1255  -3.8484  -3.2599   3.3273   0.3482
#>    1.1795  -2.6154  -5.1697   0.8964 -19.1918   0.4983   2.8026  -9.4135
#>    0.5961  -0.7496  -2.5339  -2.2819  17.2797   5.0904  -0.6621   3.1699
#>   -0.0740  -7.7036   5.4124  -5.0118   4.6653   4.9357   2.4880  -5.7561
#>    7.9524   1.2055   8.6175  -4.3379   0.8589  -5.2085  -3.8489  -1.1571
#>   -5.0426  -4.5730   0.9169   2.7226  -0.6832   4.7579   2.8094   7.0612
#>    6.4622   1.9894  -3.0439 -13.1050   3.5274  -5.7679   7.8862   7.6009
#>    4.5171  -0.4625   4.5209 -13.0029  -2.8514  -0.3041  -0.0117  -1.6034
#>    1.5260   0.5371  -0.7540   5.2496   3.7940 -15.1324  -5.3956   6.1296
#>    1.1395  -4.3057  -6.1551   2.7532   8.6251  -6.2884   7.7863  -3.4409
#>   -1.3670  -2.2836   1.4657  -0.8896   6.6559   4.2206  -8.0044  -0.2685
#>    6.8210  -6.4145  -1.4470  -0.0222 -10.2942  -6.2839   7.0761 -11.8587
#>    5.3414   6.3936   7.0699  -3.5921 -10.0294  -5.6204  11.8483  -0.4963
#>   -3.9650   5.4608   2.6019   1.9944  -9.7518   1.7764  -0.7848  -6.0065
#>   -2.2119   5.0335   5.9111   5.8334   7.0372  -3.8627  10.0256   4.2101
#>   11.2848   5.6419   4.4282   8.7233  -2.0677  13.2901  -1.8581  12.9780
#>   -2.5387   3.3702  -3.5456  -0.0669   8.8669  -3.7522   7.4880   9.3948
#>    6.0125  -4.9540   0.0705   1.0826  20.6295   7.8005   3.4489   3.2839
#>   -5.0764 -11.3325  -1.1164   8.1030   1.8788  -1.7046   0.2077  -0.6192
#>    0.9016   3.0802   7.8097  -2.6657   5.3589   6.5474   4.1099   3.1267
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
