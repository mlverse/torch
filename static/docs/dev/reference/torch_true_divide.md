# TRUE_divide

TRUE_divide

## Usage

``` r
torch_true_divide(self, other)
```

## Arguments

- self:

  (Tensor) the dividend

- other:

  (Tensor or Scalar) the divisor

## true_divide(dividend, divisor) -\> Tensor

Performs "true division" that always computes the division in floating
point. Analogous to division in Python 3 and equivalent to
[`torch_div`](https://torch.mlverse.org/docs/dev/reference/torch_div.md)
except when both inputs have bool or integer scalar types, in which case
they are cast to the default (floating) scalar type before the division.

\$\$ \mbox{out}\_i = \frac{\mbox{dividend}\_i}{\mbox{divisor}} \$\$

## Examples

``` r
if (torch_is_installed()) {

dividend = torch_tensor(c(5, 3), dtype=torch_int())
divisor = torch_tensor(c(3, 2), dtype=torch_int())
torch_true_divide(dividend, divisor)
torch_true_divide(dividend, 2)
}
#> torch_tensor
#>  2.5000
#>  1.5000
#> [ CPUFloatType{2} ]
```
