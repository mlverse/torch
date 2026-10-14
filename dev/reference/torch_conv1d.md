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
#> Columns 1 to 8   7.5675  -3.1280   0.6436  -8.6461  -7.9260   9.3685  14.9731 -14.4471
#>   17.7206 -10.9623  15.9238  12.9403   2.6358   0.7302  -2.0350  -7.4005
#>    5.5131  -5.5629  -6.6923  14.2364 -12.3252   7.3104   2.9101   1.1703
#>   -3.3370  -1.7436   4.9437 -14.3825  -4.4584   8.3147  -3.4006  -0.6338
#>   11.4001  -0.2793   5.8267  12.8004   9.0338  11.2051   3.7672  -2.2987
#>    5.4406   4.1214  11.3864  -7.2432   2.2247  -1.0108   7.2412  -8.4839
#>    5.7409  -1.5844   6.5405   0.5318  -2.7226   1.6775  -4.1859  -4.2406
#>    2.3648  -9.2087  -2.6089  -3.6547  -3.2230   1.8646   4.4304  -3.6072
#>   15.0608   0.4532   7.6450  19.4121  -2.3800  -1.2695   2.1396  -6.4578
#>    6.7893 -12.0011   3.4746   0.6672   5.9120  -0.7164   6.3470   3.0190
#>   -0.2129   2.8253  -6.3162  -0.1930  -3.7807   8.1742  -4.3058  -9.8750
#>    6.9286  -1.1832   1.7896   5.4637  13.3996  -6.7184  -2.7732 -11.2940
#>  -10.1185  -1.6622   0.2182  -7.9613  -8.0352 -11.8658  -9.4912  -1.0945
#>    2.7794  -2.4368   4.9244 -15.2067  -6.8042 -11.8254  -8.6169   2.5408
#>  -11.9823   8.6456   1.9421  -4.9012   8.0519  16.4066   5.1311  -0.4226
#>    9.8189  -9.3667   1.5132   4.8515   2.8467   4.8533  16.9567   0.2029
#>  -12.5133  13.3445 -10.4376 -11.7073   5.5365   3.4342  -0.8107   7.6150
#>  -10.8875  -4.5855   4.7747   2.1178  -9.4745  -4.5746  -0.0437  -8.4714
#>   11.1101   2.3294  -2.7320   3.6407   1.4356   0.5836   1.2062   3.5535
#>    3.5466   0.7944 -19.5710 -13.6947 -11.5063   5.8714   6.4377   2.8680
#>   -3.6928 -11.0104  -0.1207 -10.7193   3.8052   5.6889  -2.2342   1.7859
#>   -6.2647  -1.4715  -3.5792   2.6230  -4.4715  -2.2256   3.8485   9.9655
#>   -1.8225  15.2493 -14.4349 -11.9868  -3.3988  -9.0468  -3.3673  10.8387
#>   -3.6877   7.5693  12.5955   1.6804   2.1865   2.4247  -5.5495  -7.2922
#>    1.6417  -6.0326  15.5650   3.0679  -1.3162  -5.0387 -12.9940  -1.8140
#>   -8.4341  -3.0915  -0.5567  -1.1566 -13.3821  -2.7710   4.3384   8.4907
#>    0.5054  -1.7059  10.0274 -10.2176  -7.3925 -12.3049  -6.4806   3.5886
#>    3.4022  -4.6658  -1.5914 -11.1548  -1.2342  -4.2837   6.0816  -7.6028
#>   14.9436   5.3034   2.8337  -3.9810   0.0729   5.4664   0.7272 -13.5006
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
