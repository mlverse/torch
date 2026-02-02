# Hardtanh module

Applies the HardTanh function element-wise HardTanh is defined as:

## Usage

``` r
nn_hardtanh(min_val = -1, max_val = 1, inplace = FALSE)
```

## Arguments

- min_val:

  minimum value of the linear region range. Default: -1

- max_val:

  maximum value of the linear region range. Default: 1

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Details

\$\$ \mbox{HardTanh}(x) = \left\\ \begin{array}{ll} 1 & \mbox{ if } x \>
1 \\ -1 & \mbox{ if } x \< -1 \\ x & \mbox{ otherwise } \\ \end{array}
\right. \$\$

The range of the linear region :math:`[-1, 1]` can be adjusted using
`min_val` and `max_val`.

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_hardtanh(-2, 2)
input <- torch_randn(2)
output <- m(input)
}
```
