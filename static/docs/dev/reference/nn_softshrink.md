# Softshrink module

Applies the soft shrinkage function elementwise:

## Usage

``` r
nn_softshrink(lambd = 0.5)
```

## Arguments

- lambd:

  the \\\lambda\\ (must be no less than zero) value for the Softshrink
  formulation. Default: 0.5

## Details

\$\$ \mbox{SoftShrinkage}(x) = \left\\ \begin{array}{ll} x - \lambda, &
\mbox{ if } x \> \lambda \\ x + \lambda, & \mbox{ if } x \< -\lambda \\
0, & \mbox{ otherwise } \end{array} \right. \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_softshrink()
input <- torch_randn(2)
output <- m(input)
}
```
