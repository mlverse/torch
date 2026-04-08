# Flattens a contiguous range of dims into a tensor.

For use with
[nn_sequential](https://torch.mlverse.org/docs/dev/reference/nn_sequential.md).

## Usage

``` r
nn_flatten(start_dim = 2, end_dim = -1)
```

## Arguments

- start_dim:

  first dim to flatten (default = 2).

- end_dim:

  last dim to flatten (default = -1).

## Shape

- Input: `(*, S_start,..., S_i, ..., S_end, *)`, where `S_i` is the size
  at dimension `i` and `*` means any number of dimensions including
  none.

- Output: `(*, S_start*...*S_i*...S_end, *)`.

## See also

[nn_unflatten](https://torch.mlverse.org/docs/dev/reference/nn_unflatten.md)

## Examples

``` r
if (torch_is_installed()) {
input <- torch_randn(32, 1, 5, 5)
m <- nn_flatten()
m(input)
}
```
