# Hardswish module

Applies the hardswish function, element-wise, as described in the paper:
[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

## Usage

``` r
nn_hardswish()
```

## Details

\$\$ \mbox{Hardswish}(x) = \left\\ \begin{array}{ll} 0 & \mbox{if } x
\le -3, \\ x & \mbox{if } x \ge +3, \\ x \cdot (x + 3)/6 &
\mbox{otherwise} \end{array} \right. \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
m <- nn_hardswish()
input <- torch_randn(2)
output <- m(input)
} # }

}
```
