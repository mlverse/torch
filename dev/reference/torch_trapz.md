# Trapz

Trapz

## Usage

``` r
torch_trapz(y, dx = 1L, x, dim = -1L)
```

## Arguments

- y:

  (Tensor) The values of the function to integrate

- dx:

  (float) The distance between points at which `y` is sampled.

- x:

  (Tensor) The points at which the function `y` is sampled. If `x` is
  not in ascending order, intervals on which it is decreasing contribute
  negatively to the estimated integral (i.e., the convention \\\int_a^b
  f = -\int_b^a f\\ is followed).

- dim:

  (int) The dimension along which to integrate. By default, use the last
  dimension.

## trapz(y, x, \*, dim=-1) -\> Tensor

Estimate \\\int y\\dx\\ along `dim`, using the trapezoid rule.

## trapz(y, \*, dx=1, dim=-1) -\> Tensor

As above, but the sample points are spaced uniformly at a distance of
`dx`.

## Examples

``` r
if (torch_is_installed()) {

y = torch_randn(list(2, 3))
y
x = torch_tensor(matrix(c(1, 3, 4, 1, 2, 3), ncol = 3, byrow=TRUE))
torch_trapz(y, x = x)

}
#> torch_tensor
#>  0.6193
#>  0.1711
#> [ CPUFloatType{2} ]
```
