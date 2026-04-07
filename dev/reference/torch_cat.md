# Cat

Cat

## Usage

``` r
torch_cat(tensors, dim = 1L)
```

## Arguments

- tensors:

  (sequence of Tensors) any python sequence of tensors of the same type.
  Non-empty tensors provided must have the same shape, except in the cat
  dimension.

- dim:

  (int, optional) the dimension over which the tensors are concatenated

## cat(tensors, dim=0, out=NULL) -\> Tensor

Concatenates the given sequence of `seq` tensors in the given dimension.
All tensors must either have the same shape (except in the concatenating
dimension) or be empty.

`torch_cat` can be seen as an inverse operation for
[`torch_split()`](https://torch.mlverse.org/docs/dev/reference/torch_split.md)
and
[`torch_chunk`](https://torch.mlverse.org/docs/dev/reference/torch_chunk.md).

`torch_cat` can be best understood via examples.

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(2, 3))
x
torch_cat(list(x, x, x), 1)
torch_cat(list(x, x, x), 2)
}
#> torch_tensor
#>  1.0107 -0.7715 -0.4267  1.0107 -0.7715 -0.4267  1.0107 -0.7715 -0.4267
#>  0.0950  0.2772  0.1993  0.0950  0.2772  0.1993  0.0950  0.2772  0.1993
#> [ CPUFloatType{2,9} ]
```
