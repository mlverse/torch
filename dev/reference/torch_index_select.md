# Index_select

Index_select

## Usage

``` r
torch_index_select(self, dim, index)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension in which we index

- index:

  (LongTensor) the 1-D tensor containing the indices to index

## Note

The returned tensor does **not** use the same storage as the original
tensor. If `out` has a different shape than expected, we silently change
it to the correct shape, reallocating the underlying storage if
necessary.

## index_select(input, dim, index, out=NULL) -\> Tensor

Returns a new tensor which indexes the `input` tensor along dimension
`dim` using the entries in `index` which is a `LongTensor`.

The returned tensor has the same number of dimensions as the original
tensor (`input`). The `dim`\\ th dimension has the same size as the
length of `index`; other dimensions have the same size as in the
original tensor.

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(3, 4))
x
indices = torch_tensor(c(1, 3), dtype = torch_int64())
torch_index_select(x, 1, indices)
torch_index_select(x, 2, indices)
}
#> torch_tensor
#>  1.6534 -1.3725
#> -0.7858  0.7047
#>  0.6983 -1.1160
#> [ CPUFloatType{3,2} ]
```
