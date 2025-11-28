# Clamp

Clamp

## Usage

``` r
torch_clamp(self, min = NULL, max = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- min:

  (Number) lower-bound of the range to be clamped to

- max:

  (Number) upper-bound of the range to be clamped to

## clamp(input, min, max, out=NULL) -\> Tensor

Clamp all elements in `input` into the range `[` `min`, `max` `]` and
return a resulting tensor:

\$\$ y_i = \left\\ \begin{array}{ll} \mbox{min} & \mbox{if } x_i \<
\mbox{min} \\ x_i & \mbox{if } \mbox{min} \leq x_i \leq \mbox{max} \\
\mbox{max} & \mbox{if } x_i \> \mbox{max} \end{array} \right. \$\$ If
`input` is of type `FloatTensor` or `DoubleTensor`, args `min` and `max`
must be real numbers, otherwise they should be integers.

## clamp(input, \*, min, out=NULL) -\> Tensor

Clamps all elements in `input` to be larger or equal `min`.

If `input` is of type `FloatTensor` or `DoubleTensor`, `value` should be
a real number, otherwise it should be an integer.

## clamp(input, \*, max, out=NULL) -\> Tensor

Clamps all elements in `input` to be smaller or equal `max`.

If `input` is of type `FloatTensor` or `DoubleTensor`, `value` should be
a real number, otherwise it should be an integer.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_clamp(a, min=-0.5, max=0.5)


a = torch_randn(c(4))
a
torch_clamp(a, min=0.5)


a = torch_randn(c(4))
a
torch_clamp(a, max=0.5)
}
#> torch_tensor
#> -1.3410
#>  0.5000
#>  0.2962
#> -0.0662
#> [ CPUFloatType{4} ]
```
