# Elu

Applies element-wise, \$\$ELU(x) = max(0,x) + min(0, \alpha \* (exp(x) -
1))\$\$.

## Usage

``` r
nnf_elu(input, alpha = 1, inplace = FALSE)

nnf_elu_(input, alpha = 1)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- alpha:

  the alpha value for the ELU formulation. Default: 1.0

- inplace:

  can optionally do the operation in-place. Default: FALSE

## Examples

``` r
if (torch_is_installed()) {
x <- torch_randn(2, 2)
y <- nnf_elu(x, alpha = 1)
nnf_elu_(x, alpha = 1)
torch_equal(x, y)
}
#> [1] TRUE
```
