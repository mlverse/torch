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
#> -0.1175  0.6475  1.4589 -0.7428
#> -1.9019 -1.1369 -0.3256 -2.5273
#> -0.1378  0.6272  1.4386 -0.7631
#> -0.0787  0.6863  1.4976 -0.7041
#> [ CPUFloatType{4,4} ]
```
