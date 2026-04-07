# Add

Add

## Usage

``` r
torch_add(self, other, alpha = 1L)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor/Number) the second input tensor/number.

- alpha:

  (Number) the scalar multiplier for `other`

## add(input, other, out=NULL)

Adds the scalar `other` to each element of the input `input` and returns
a new resulting tensor.

\$\$ \mbox{out} = \mbox{input} + \mbox{other} \$\$ If `input` is of type
FloatTensor or DoubleTensor, `other` must be a real number, otherwise it
should be an integer.

## add(input, other, \*, alpha=1, out=NULL)

Each element of the tensor `other` is multiplied by the scalar `alpha`
and added to each element of the tensor `input`. The resulting tensor is
returned.

The shapes of `input` and `other` must be broadcastable .

\$\$ \mbox{out} = \mbox{input} + \mbox{alpha} \times \mbox{other} \$\$
If `other` is of type FloatTensor or DoubleTensor, `alpha` must be a
real number, otherwise it should be an integer.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_add(a, 20)


a = torch_randn(c(4))
a
b = torch_randn(c(4, 1))
b
torch_add(a, b)
}
#> torch_tensor
#> -2.5024 -1.4334 -0.8979 -1.1497
#> -1.3805 -0.3115  0.2240 -0.0278
#> -3.2612 -2.1923 -1.6568 -1.9086
#> -2.0193 -0.9503 -0.4148 -0.6666
#> [ CPUFloatType{4,4} ]
```
