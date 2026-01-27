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
#>  2.1728e+00  1.7133e-03  8.9354e-01 -4.8333e-01
#> -9.7673e-01 -7.7019e-04 -4.0167e-01  2.1727e-01
#>  3.3764e+00  2.6624e-03  1.3885e+00 -7.5107e-01
#> -6.3065e-02 -4.9729e-05 -2.5935e-02  1.4029e-02
#> [ CPUFloatType{4,4} ]
```
