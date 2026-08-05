# GELU module

Applies the Gaussian Error Linear Units function: \$\$\mbox{GELU}(x) = x
\* \Phi(x)\$\$

## Usage

``` r
nn_gelu(approximate = "none")
```

## Arguments

- approximate:

  the gelu approximation algorithm to use: `'none'` or `'tanh'`.
  Default: `'none'`.

## Details

where \\\Phi(x)\\ is the Cumulative Distribution Function for Gaussian
Distribution.

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_gelu()
input <- torch_randn(2)
output <- m(input)
}
```
