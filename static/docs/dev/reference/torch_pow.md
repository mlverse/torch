# Pow

Pow

## Usage

``` r
torch_pow(self, exponent)
```

## Arguments

- self:

  (float) the scalar base value for the power operation

- exponent:

  (float or tensor) the exponent value

## pow(input, exponent, out=NULL) -\> Tensor

Takes the power of each element in `input` with `exponent` and returns a
tensor with the result.

`exponent` can be either a single `float` number or a `Tensor` with the
same number of elements as `input`.

When `exponent` is a scalar value, the operation applied is:

\$\$ \mbox{out}\_i = x_i^{\mbox{exponent}} \$\$ When `exponent` is a
tensor, the operation applied is:

\$\$ \mbox{out}\_i = x_i^{\mbox{exponent}\_i} \$\$ When `exponent` is a
tensor, the shapes of `input` and `exponent` must be broadcastable .

## pow(self, exponent, out=NULL) -\> Tensor

`self` is a scalar `float` value, and `exponent` is a tensor. The
returned tensor `out` is of the same shape as `exponent`

The operation applied is:

\$\$ \mbox{out}\_i = \mbox{self} ^ {\mbox{exponent}\_i} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_pow(a, 2)
exp <- torch_arange(1, 5)
a <- torch_arange(1, 5)
a
exp
torch_pow(a, exp)


exp <- torch_arange(1, 5)
base <- 2
torch_pow(base, exp)
}
#> torch_tensor
#>   2
#>   4
#>   8
#>  16
#>  32
#> [ CPUFloatType{5} ]
```
