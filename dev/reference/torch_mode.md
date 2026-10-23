# Mode

Mode

## Usage

``` r
torch_mode(self, dim = -1L, keepdim = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

## Note

This function is not defined for `torch_cuda.Tensor` yet.

## mode(input, dim=-1, keepdim=False, out=NULL) -\> (Tensor, LongTensor)

Returns a namedtuple `(values, indices)` where `values` is the mode
value of each row of the `input` tensor in the given dimension `dim`,
i.e. a value which appears most often in that row, and `indices` is the
index location of each mode value found.

By default, `dim` is the last dimension of the `input` tensor.

If `keepdim` is `TRUE`, the output tensors are of the same size as
`input` except in the dimension `dim` where they are of size 1.
Otherwise, `dim` is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensors having 1 fewer dimension than `input`.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randint(0, 50, size = list(5))
a
torch_mode(a, 1)
}
#> [[1]]
#> torch_tensor
#> 7
#> [ CPUFloatType{} ]
#> 
#> [[2]]
#> torch_tensor
#> 0
#> [ CPULongType{} ]
#> 
```
