# Hardshwink module

Applies the hard shrinkage function element-wise:

## Usage

``` r
nn_hardshrink(lambd = 0.5)
```

## Arguments

- lambd:

  the \\\lambda\\ value for the Hardshrink formulation. Default: 0.5

## Details

\$\$ \mbox{HardShrink}(x) = \left\\ \begin{array}{ll} x, & \mbox{ if } x
\> \lambda \\ x, & \mbox{ if } x \< -\lambda \\ 0, & \mbox{ otherwise }
\end{array} \right. \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_hardshrink()
input <- torch_randn(2)
output <- m(input)
}
```
