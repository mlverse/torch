# As_strided

As_strided

## Usage

``` r
torch_as_strided(self, size, stride, storage_offset = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- size:

  (tuple or ints) the shape of the output tensor

- stride:

  (tuple or ints) the stride of the output tensor

- storage_offset:

  (int, optional) the offset in the underlying storage of the output
  tensor

## as_strided(input, size, stride, storage_offset=0) -\> Tensor

Create a view of an existing `torch_Tensor` `input` with specified
`size`, `stride` and `storage_offset`.

## Warning

More than one element of a created tensor may refer to a single memory
location. As a result, in-place operations (especially ones that are
vectorized) may result in incorrect behavior. If you need to write to
the tensors, please clone them first.

    Many PyTorch functions, which return a view of a tensor, are internally
    implemented with this function. Those functions, like
    `torch_Tensor.expand`, are easier to read and are therefore more
    advisable to use.

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(3, 3))
x
t = torch_as_strided(x, list(2, 2), list(1, 2))
t
t = torch_as_strided(x, list(2, 2), list(1, 2), 1)
t
}
#> torch_tensor
#> -1.6077 -1.7359
#> -0.8358 -1.0812
#> [ CPUFloatType{2,2} ]
```
