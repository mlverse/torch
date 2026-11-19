# Hardswish

Applies the hardswish function, element-wise, as described in the paper:
Searching for MobileNetV3.

## Usage

``` r
nnf_hardswish(input, inplace = FALSE)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- inplace:

  can optionally do the operation in-place. Default: FALSE

## Details

\$\$ \mbox{Hardswish}(x) = \left\\ \begin{array}{ll} 0 & \mbox{if } x
\le -3, \\ x & \mbox{if } x \ge +3, \\ x \cdot (x + 3)/6 &
\mbox{otherwise} \end{array} \right. \$\$
