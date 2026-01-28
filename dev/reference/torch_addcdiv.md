# Addcdiv

Addcdiv

## Usage

``` r
torch_addcdiv(self, tensor1, tensor2, value = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to be added

- tensor1:

  (Tensor) the numerator tensor

- tensor2:

  (Tensor) the denominator tensor

- value:

  (Number, optional) multiplier for \\\mbox{tensor1} / \mbox{tensor2}\\

## addcdiv(input, tensor1, tensor2, \*, value=1, out=NULL) -\> Tensor

Performs the element-wise division of `tensor1` by `tensor2`, multiply
the result by the scalar `value` and add it to `input`.

## Warning

Integer division with addcdiv is deprecated, and in a future release
addcdiv will perform a true division of `tensor1` and `tensor2`. The
current addcdiv behavior can be replicated using
[`torch_floor_divide()`](https://torch.mlverse.org/docs/dev/reference/torch_floor_divide.md)
for integral inputs (`input` + `value` \* `tensor1` // `tensor2`) and
[`torch_div()`](https://torch.mlverse.org/docs/dev/reference/torch_div.md)
for float inputs (`input` + `value` \* `tensor1` / `tensor2`). The new
addcdiv behavior can be implemented with
[`torch_true_divide()`](https://torch.mlverse.org/docs/dev/reference/torch_true_divide.md)
(`input` + `value` \* torch.true_divide(`tensor1`, `tensor2`).

\$\$ \mbox{out}\_i = \mbox{input}\_i + \mbox{value} \times
\frac{\mbox{tensor1}\_i}{\mbox{tensor2}\_i} \$\$

The shapes of `input`, `tensor1`, and `tensor2` must be broadcastable .

For inputs of type `FloatTensor` or `DoubleTensor`, `value` must be a
real number, otherwise an integer.

## Examples

``` r
if (torch_is_installed()) {

t = torch_randn(c(1, 3))
t1 = torch_randn(c(3, 1))
t2 = torch_randn(c(1, 3))
torch_addcdiv(t, t1, t2, 0.1)
}
#> torch_tensor
#>  0.3763  0.1076 -1.1792
#>  0.4747  0.3953 -0.9754
#>  0.1855 -0.4502 -1.5743
#> [ CPUFloatType{3,3} ]
```
