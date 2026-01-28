# Hardsigmoid module

Applies the element-wise function:

## Usage

``` r
nn_hardsigmoid()
```

## Details

\$\$ \mbox{Hardsigmoid}(x) = \left\\ \begin{array}{ll} 0 & \mbox{if~} x
\le -3, \\ 1 & \mbox{if~} x \ge +3, \\ x / 6 + 1 / 2 & \mbox{otherwise}
\end{array} \right. \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_hardsigmoid()
input <- torch_randn(2)
output <- m(input)
}
```
