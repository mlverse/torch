# Mul

Mul

## Usage

``` r
torch_mul(self, other)
```

## Arguments

- self:

  (Tensor) the first multiplicand tensor

- other:

  (Tensor) the second multiplicand tensor

## mul(input, other, out=NULL)

Multiplies each element of the input `input` with the scalar `other` and
returns a new resulting tensor.

\$\$ \mbox{out}\_i = \mbox{other} \times \mbox{input}\_i \$\$ If `input`
is of type `FloatTensor` or `DoubleTensor`, `other` should be a real
number, otherwise it should be an integer

Each element of the tensor `input` is multiplied by the corresponding
element of the Tensor `other`. The resulting tensor is returned.

The shapes of `input` and `other` must be broadcastable .

\$\$ \mbox{out}\_i = \mbox{input}\_i \times \mbox{other}\_i \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(3))
a
torch_mul(a, 100)


a = torch_randn(c(4, 1))
a
b = torch_randn(c(1, 4))
b
torch_mul(a, b)
}
#> torch_tensor
#> -0.0699  2.9616  1.1445  0.7828
#>  0.0167 -0.7079 -0.2736 -0.1871
#>  0.0508 -2.1512 -0.8313 -0.5686
#> -0.0064  0.2728  0.1054  0.0721
#> [ CPUFloatType{4,4} ]
```
