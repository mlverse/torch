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
#> Columns 1 to 8   3.8450  -1.5341   3.2991  -1.6388  -2.8139   1.5471  -0.2098   5.7365
#>   -1.3499  -0.1530   1.6461   5.6442  -2.0061   0.4074  -6.2149   3.1649
#>    1.1804   2.2060  -3.0091  -8.4105  -9.2602  -0.0075   6.6166  -0.0372
#>    2.0760  -1.3783   4.6561   4.7727  -8.1210   6.5215   6.3307 -16.8271
#>    8.4066   5.6322   5.8001  -4.0988  -2.6091   2.0914   9.6207  -3.6803
#>    0.4589  -3.9337  -9.4639  -3.8243   0.2849  10.0828  -2.5770   9.6965
#>    4.4080   4.3055   3.4169   5.5801   3.9841  -0.8147  10.4547   1.0024
#>   -1.0612  -1.2395   1.2390   2.8196  -3.1740  -4.4583   8.5225  -4.7603
#>    8.7276   1.7149  -9.1262  -4.3852  -5.3765 -13.2662  11.1171 -17.3590
#>   -6.3841  -0.8157   4.3438   8.4130   2.1375  -0.6758 -11.2861  12.9668
#>   -4.5571  -3.0960   2.2884  -6.2091  -6.7904   0.0073   1.1299  -7.1733
#>    2.8583  14.3724  19.9983   2.4880   3.1753  -0.6159   2.2275  -1.0133
#>   -0.2901   8.7701  -1.2729  -4.6080   2.7192   4.7325   1.6007   5.2146
#>   -3.7041   3.0247  12.6400  -1.5522  -5.5725   5.7301  -8.5288   9.3195
#>   -4.4418  -8.2316  -6.2868  -9.1847  -2.9181   1.4809  -8.2908   2.0026
#>    1.3031   7.4962  -9.5737  -1.8063  11.9649   5.7917   3.4704  -4.0552
#>   -6.1957  -0.2990   9.7631   2.5652   1.6673  -3.7618  -6.3822 -11.1550
#>    3.5758   4.2067   0.3680  -7.6746   0.6337  12.1423  15.0068   1.9558
#>    0.3535  -7.1274   5.7309   3.1889  -0.7891   1.7460   4.9770  -3.1754
#>    2.6634  -1.7734  -1.9581   2.5256  14.7535   7.5878  -2.8931   1.9102
#>   -1.9842  -1.6679   1.9234   8.1594   2.1125   9.2329   8.1303   8.7554
#>    9.9370   2.8510   5.5238   5.6991   3.2581  13.0484  -0.0744  -2.9656
#>   11.3374  -5.0111  -7.7053  -6.9006 -10.0976   2.2513  -0.5489  -3.1487
#>   -1.6663  10.3526   1.6276  -6.7372  -1.6337   3.8976  -4.7769   3.4530
#>    1.0308   2.2257   3.9345  -8.2579  -1.9767   5.4078   2.1882   3.1881
#>    2.6116  -7.3549  -6.8855  18.2375   8.8959   3.3855   0.4526   3.2569
#>    2.6978 -10.7726   4.2770  -2.7295  -3.0820   2.3624  -1.0260  -8.6441
#>    4.1990  -1.6911  -7.7386  -0.6395   0.9778  12.4755   2.1052   1.7473
#>   -3.6157  -9.8974  -4.8812  -9.0784   7.0850 -15.9082   2.5604  -9.0876
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
