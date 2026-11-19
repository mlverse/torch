# Hypot

Hypot

## Usage

``` r
torch_hypot(self, other)
```

## Arguments

- self:

  (Tensor) the first input tensor

- other:

  (Tensor) the second input tensor

## hypot(input, other, \*, out=None) -\> Tensor

Given the legs of a right triangle, return its hypotenuse.

\$\$ \mbox{out}\_{i} = \sqrt{\mbox{input}\_{i}^{2} +
\mbox{other}\_{i}^{2}} \$\$

The shapes of `input` and `other` must be broadcastable .

## Examples

``` r
if (torch_is_installed()) {

torch_hypot(torch_tensor(c(4.0)), torch_tensor(c(3.0, 4.0, 5.0)))
}
#> torch_tensor
#>  5.0000
#>  5.6569
#>  6.4031
#> [ CPUFloatType{3} ]
```
