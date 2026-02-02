# Unflattens a tensor dim expanding it to a desired shape. For use with \[[nn_sequential](https://torch.mlverse.org/docs/dev/reference/nn_sequential.md).

Unflattens a tensor dim expanding it to a desired shape. For use with
\[[nn_sequential](https://torch.mlverse.org/docs/dev/reference/nn_sequential.md).

## Usage

``` r
nn_unflatten(dim, unflattened_size)
```

## Arguments

- dim:

  Dimension to be unflattened

- unflattened_size:

  New shape of the unflattened dimension

## Examples

``` r
if (torch_is_installed()) {
input <- torch_randn(2, 50)

m <- nn_sequential(
  nn_linear(50, 50),
  nn_unflatten(2, c(2, 5, 5))
)
output <- m(input)
output$size()
}
#> [1] 2 2 5 5
```
