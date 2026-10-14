# Tensordot

Returns a contraction of a and b over multiple dimensions. `tensordot`
implements a generalized matrix product.

## Usage

``` r
torch_tensordot(a, b, dims = 2)
```

## Arguments

- a:

  (Tensor) Left tensor to contract

- b:

  (Tensor) Right tensor to contract

- dims:

  (int or tuple of two lists of integers) number of dimensions to
  contract or explicit lists of dimensions for `a` and `b` respectively

## Examples

``` r
if (torch_is_installed()) {

a <- torch_arange(start = 1, end = 60)$reshape(c(3, 4, 5))
b <- torch_arange(start = 1, end = 24)$reshape(c(4, 3, 2))
torch_tensordot(a, b, dims = list(c(2, 1), c(1, 2)))
if (FALSE) { # \dontrun{
a = torch_randn(3, 4, 5, device='cuda')
b = torch_randn(4, 5, 6, device='cuda')
c = torch_tensordot(a, b, dims=2)$cpu()
} # }
}
```
