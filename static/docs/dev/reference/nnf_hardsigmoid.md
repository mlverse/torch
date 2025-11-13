# Hardsigmoid

Applies the element-wise function \\\mbox{Hardsigmoid}(x) =
\frac{ReLU6(x + 3)}{6}\\

## Usage

``` r
nnf_hardsigmoid(input, inplace = FALSE)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- inplace:

  NA If set to `True`, will do this operation in-place. Default: `False`
