# Logaddexp2

Logaddexp2

## Usage

``` r
torch_logaddexp2(self, other)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the second input tensor

## logaddexp2(input, other, \*, out=None) -\> Tensor

Logarithm of the sum of exponentiations of the inputs in base-2.

Calculates pointwise \\\log_2\left(2^x + 2^y\right)\\. See
[`torch_logaddexp()`](https://torch.mlverse.org/docs/dev/reference/torch_logaddexp.md)
for more details.
