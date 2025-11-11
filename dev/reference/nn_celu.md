# CELU module

Applies the element-wise function:

## Usage

``` r
nn_celu(alpha = 1, inplace = FALSE)
```

## Arguments

- alpha:

  the \\\alpha\\ value for the CELU formulation. Default: 1.0

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Details

\$\$ \mbox{CELU}(x) = \max(0,x) + \min(0, \alpha \* (\exp(x/\alpha) -
1)) \$\$

More details can be found in the paper [Continuously Differentiable
Exponential Linear Units](https://arxiv.org/abs/1704.07483).

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_celu()
input <- torch_randn(2)
output <- m(input)
}
```
