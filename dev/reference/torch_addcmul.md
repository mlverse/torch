# Addcmul

Addcmul

## Usage

``` r
torch_addcmul(self, tensor1, tensor2, value = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to be added

- tensor1:

  (Tensor) the tensor to be multiplied

- tensor2:

  (Tensor) the tensor to be multiplied

- value:

  (Number, optional) multiplier for \\tensor1 .\* tensor2\\

## addcmul(input, tensor1, tensor2, \*, value=1, out=NULL) -\> Tensor

Performs the element-wise multiplication of `tensor1` by `tensor2`,
multiply the result by the scalar `value` and add it to `input`.

\$\$ \mbox{out}\_i = \mbox{input}\_i + \mbox{value} \times
\mbox{tensor1}\_i \times \mbox{tensor2}\_i \$\$ The shapes of `tensor`,
`tensor1`, and `tensor2` must be broadcastable .

For inputs of type `FloatTensor` or `DoubleTensor`, `value` must be a
real number, otherwise an integer.

## Examples

``` r
if (torch_is_installed()) {

t = torch_randn(c(1, 3))
t1 = torch_randn(c(3, 1))
t2 = torch_randn(c(1, 3))
torch_addcmul(t, t1, t2, 0.1)
}
#> torch_tensor
#> -0.2449 -0.8711 -0.2466
#> -0.2389 -0.9152 -0.2624
#> -0.2526 -0.8152 -0.2266
#> [ CPUFloatType{3,3} ]
```
