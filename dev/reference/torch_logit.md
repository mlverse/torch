# Logit

Logit

## Usage

``` r
torch_logit(self, eps = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- eps:

  (float, optional) the epsilon for input clamp bound. Default: `None`

## logit(input, eps=None, \*, out=None) -\> Tensor

Returns a new tensor with the logit of the elements of `input`. `input`
is clamped to `[eps, 1 - eps]` when eps is not None. When eps is None
and `input` \< 0 or `input` \> 1, the function will yields NaN.

\$\$ y\_{i} = \ln(\frac{z\_{i}}{1 - z\_{i}}) \\ z\_{i} =
\begin{array}{ll} x\_{i} & \mbox{if eps is None} \\ \mbox{eps} &
\mbox{if } x\_{i} \< \mbox{eps} \\ x\_{i} & \mbox{if } \mbox{eps} \leq
x\_{i} \leq 1 - \mbox{eps} \\ 1 - \mbox{eps} & \mbox{if } x\_{i} \> 1 -
\mbox{eps} \end{array} \$\$

## Examples

``` r
if (torch_is_installed()) {

a <- torch_rand(5)
a
torch_logit(a, eps=1e-6)
}
#> torch_tensor
#> -0.1032
#>  1.5252
#> -1.0741
#>  0.9972
#>  0.0254
#> [ CPUFloatType{5} ]
```
