# Max

Max

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not. Default:
  `FALSE`.

- out:

  (tuple, optional) the result tuple of two output tensors (max,
  max_indices)

- other:

  (Tensor) the second input tensor

## Note

When the shapes do not match, the shape of the returned output tensor
follows the broadcasting rules .

## max(input) -\> Tensor

Returns the maximum value of all elements in the `input` tensor.

## max(input, dim, keepdim=False, out=NULL) -\> (Tensor, LongTensor)

Returns a namedtuple `(values, indices)` where `values` is the maximum
value of each row of the `input` tensor in the given dimension `dim`.
And `indices` is the index location of each maximum value found
(argmax).

## Warning

`indices` does not necessarily contain the first occurrence of each
maximal value found, unless it is unique. The exact implementation
details are device-specific. Do not expect the same result when run on
CPU and GPU in general.

If `keepdim` is `TRUE`, the output tensors are of the same size as
`input` except in the dimension `dim` where they are of size 1.
Otherwise, `dim` is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensors having 1 fewer dimension than `input`.

## max(input, other, out=NULL) -\> Tensor

Each element of the tensor `input` is compared with the corresponding
element of the tensor `other` and an element-wise maximum is taken.

The shapes of `input` and `other` don't need to match, but they must be
broadcastable .

\$\$ \mbox{out}\_i = \max(\mbox{tensor}\_i, \mbox{other}\_i) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_max(a)


a = torch_randn(c(4, 4))
a
torch_max(a, dim = 1)


a = torch_randn(c(4))
a
b = torch_randn(c(4))
b
torch_max(a, other = b)
}
#> torch_tensor
#> -0.4378
#>  0.2551
#>  0.3706
#>  1.6962
#> [ CPUFloatType{4} ]
```
