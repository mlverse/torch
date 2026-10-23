# Identity module

A placeholder identity operator that is argument-insensitive.

## Usage

``` r
nn_identity(...)
```

## Arguments

- ...:

  any arguments (unused)

## Examples

``` r
if (torch_is_installed()) {
m <- nn_identity(54, unused_argument1 = 0.1, unused_argument2 = FALSE)
input <- torch_randn(128, 20)
output <- m(input)
print(output$size())
}
#> [1] 128  20
```
