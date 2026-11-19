# Heaviside

Heaviside

## Usage

``` r
torch_heaviside(self, values)
```

## Arguments

- self:

  (Tensor) the input tensor.

- values:

  (Tensor) The values to use where `input` is zero.

## heaviside(input, values, \*, out=None) -\> Tensor

Computes the Heaviside step function for each element in `input`. The
Heaviside step function is defined as:

\$\$ \mbox{{heaviside}}(input, values) = \begin{array}{ll} 0, & \mbox{if
input \< 0}\\ values, & \mbox{if input == 0}\\ 1, & \mbox{if input \> 0}
\end{array} \$\$

## Examples

``` r
if (torch_is_installed()) {

input <- torch_tensor(c(-1.5, 0, 2.0))
values <- torch_tensor(c(0.5))
torch_heaviside(input, values)
values <- torch_tensor(c(1.2, -2.0, 3.5))
torch_heaviside(input, values)
}
#> torch_tensor
#>  0
#> -2
#>  1
#> [ CPUFloatType{3} ]
```
