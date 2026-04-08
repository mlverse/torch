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
#>  1.0466  0.4381  0.6994  0.8266
#>  2.0280  1.4195  1.6807  1.8080
#>  1.2700  0.6614  0.9227  1.0499
#>  1.6574  1.0489  1.3102  1.4374
#> [ CPUFloatType{4,4} ]
```
