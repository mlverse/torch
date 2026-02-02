# ELU module

Applies the element-wise function:

## Usage

``` r
nn_elu(alpha = 1, inplace = FALSE)
```

## Arguments

- alpha:

  the \\\alpha\\ value for the ELU formulation. Default: 1.0

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Details

\$\$ \mbox{ELU}(x) = \max(0,x) + \min(0, \alpha \* (\exp(x) - 1)) \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_elu()
input <- torch_randn(2)
output <- m(input)
}
```
