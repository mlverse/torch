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
#> -0.4200 -0.6361  0.1417 -0.9645
#>  1.8267  2.7664 -0.6163  4.1945
#> -1.0256 -1.5532  0.3460 -2.3550
#>  1.9486  2.9511 -0.6574  4.4746
#> [ CPUFloatType{4,4} ]
```
