# Equal

Equal

## Usage

``` r
torch_equal(self, other)
```

## Arguments

- self:

  the input tensor

- other:

  the other input tensor

## equal(input, other) -\> bool

`TRUE` if two tensors have the same size and elements, `FALSE`
otherwise.

## Examples

``` r
if (torch_is_installed()) {

torch_equal(torch_tensor(c(1, 2)), torch_tensor(c(1, 2)))
}
#> [1] TRUE
```
