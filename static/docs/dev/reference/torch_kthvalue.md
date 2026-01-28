# Kthvalue

Kthvalue

## Usage

``` r
torch_kthvalue(self, k, dim = -1L, keepdim = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- k:

  (int) k for the k-th smallest element

- dim:

  (int, optional) the dimension to find the kth value along

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

## kthvalue(input, k, dim=NULL, keepdim=False, out=NULL) -\> (Tensor, LongTensor)

Returns a namedtuple `(values, indices)` where `values` is the `k` th
smallest element of each row of the `input` tensor in the given
dimension `dim`. And `indices` is the index location of each element
found.

If `dim` is not given, the last dimension of the `input` is chosen.

If `keepdim` is `TRUE`, both the `values` and `indices` tensors are the
same size as `input`, except in the dimension `dim` where they are of
size 1. Otherwise, `dim` is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in both the `values` and `indices` tensors having 1 fewer
dimension than the `input` tensor.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_arange(1, 6)
x
torch_kthvalue(x, 4)
x <- torch_arange(1,6)$resize_(c(2,3))
x
torch_kthvalue(x, 2, 1, TRUE)
}
#> [[1]]
#> torch_tensor
#>  4  5  6
#> [ CPUFloatType{1,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  1  1  1
#> [ CPULongType{1,3} ]
#> 
```
