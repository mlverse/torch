# LeakyReLU module

Applies the element-wise function:

## Usage

``` r
nn_leaky_relu(negative_slope = 0.01, inplace = FALSE)
```

## Arguments

- negative_slope:

  Controls the angle of the negative slope. Default: 1e-2

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Details

\$\$ \mbox{LeakyReLU}(x) = \max(0, x) + \mbox{negative\\slope} \*
\min(0, x) \$\$ or

\$\$ \mbox{LeakyRELU}(x) = \left\\ \begin{array}{ll} x, & \mbox{ if } x
\geq 0 \\ \mbox{negative\\slope} \times x, & \mbox{ otherwise }
\end{array} \right. \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_leaky_relu(0.1)
input <- torch_randn(2)
output <- m(input)
}
```
