# Floor_divide

Floor_divide

## Usage

``` r
torch_floor_divide(self, other)
```

## Arguments

- self:

  (Tensor) the numerator tensor

- other:

  (Tensor or Scalar) the denominator

## floor_divide(input, other, out=NULL) -\> Tensor

Return the division of the inputs rounded down to the nearest integer.
See
[`torch_div`](https://torch.mlverse.org/docs/dev/reference/torch_div.md)
for type promotion and broadcasting rules.

\$\$ \mbox{{out}}\_i = \left\lfloor
\frac{{\mbox{{input}}\_i}}{{\mbox{{other}}\_i}} \right\rfloor \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_tensor(c(4.0, 3.0))
b = torch_tensor(c(2.0, 2.0))
torch_floor_divide(a, b)
torch_floor_divide(a, 1.4)
}
#> torch_tensor
#>  2
#>  2
#> [ CPUFloatType{2} ]
```
