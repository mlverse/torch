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
#>  0.0183  1.1107  0.2752  0.1861
#>  0.0063  0.3820  0.0947  0.0640
#>  0.0039  0.2366  0.0586  0.0396
#>  0.0081  0.4893  0.1213  0.0820
#> [ CPUFloatType{4,4} ]
```
