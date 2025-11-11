# Selu

Applies element-wise, \$\$SELU(x) = scale \* (max(0,x) + min(0, \alpha
\* (exp(x) - 1)))\$\$, with \\\alpha=1.6732632423543772848170429916717\\
and \\scale=1.0507009873554804934193349852946\\.

## Usage

``` r
nnf_selu(input, inplace = FALSE)

nnf_selu_(input)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- inplace:

  can optionally do the operation in-place. Default: FALSE

## Examples

``` r
if (torch_is_installed()) {
x <- torch_randn(2, 2)
y <- nnf_selu(x)
nnf_selu_(x)
torch_equal(x, y)
}
#> [1] TRUE
```
