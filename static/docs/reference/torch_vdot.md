# Vdot

Vdot

## Usage

``` r
torch_vdot(self, other)
```

## Arguments

- self:

  (Tensor) first tensor in the dot product. Its conjugate is used if
  it's complex.

- other:

  (Tensor) second tensor in the dot product.

## Note

This function does not broadcast .

## vdot(input, other, \*, out=None) -\> Tensor

Computes the dot product (inner product) of two tensors. The vdot(a, b)
function handles complex numbers differently than dot(a, b). If the
first argument is complex, the complex conjugate of the first argument
is used for the calculation of the dot product.

## Examples

``` r
if (torch_is_installed()) {

torch_vdot(torch_tensor(c(2, 3)), torch_tensor(c(2, 1)))
if (FALSE) {
a <- torch_tensor(list(1 +2i, 3 - 1i))
b <- torch_tensor(list(2 +1i, 4 - 0i))
torch_vdot(a, b)
torch_vdot(b, a)
}
}
```
