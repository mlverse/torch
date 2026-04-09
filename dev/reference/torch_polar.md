# Polar

Polar

## Usage

``` r
torch_polar(abs, angle)
```

## Arguments

- abs:

  (Tensor) The absolute value the complex tensor. Must be float or
  double.

- angle:

  (Tensor) The angle of the complex tensor. Must be same dtype as `abs`.

## polar(abs, angle, \*, out=None) -\> Tensor

Constructs a complex tensor whose elements are Cartesian coordinates
corresponding to the polar coordinates with absolute value `abs` and
angle `angle`.

\$\$ \mbox{out} = \mbox{abs} \cdot \cos(\mbox{angle}) + \mbox{abs} \cdot
\sin(\mbox{angle}) \cdot j \$\$

## Examples

``` r
if (torch_is_installed()) {

abs <- torch_tensor(c(1, 2), dtype=torch_float64())
angle <- torch_tensor(c(pi / 2, 5 * pi / 4), dtype=torch_float64())
z <- torch_polar(abs, angle)
z
}
#> torch_tensor
#> ℹ Use `$real` or `$imag` to print the contents of this tensor.
#> [ CPUComplexDoubleType{2} ]
```
