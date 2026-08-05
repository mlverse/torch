# Sparse_coo_tensor

Sparse_coo_tensor

## Usage

``` r
torch_sparse_coo_tensor(
  indices,
  values,
  size = NULL,
  dtype = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- indices:

  (array_like) Initial data for the tensor. Can be a list, tuple, NumPy
  `ndarray`, scalar, and other types. Will be cast to a
  `torch_LongTensor` internally. The indices are the coordinates of the
  non-zero values in the matrix, and thus should be two-dimensional
  where the first dimension is the number of tensor dimensions and the
  second dimension is the number of non-zero values.

- values:

  (array_like) Initial values for the tensor. Can be a list, tuple,
  NumPy `ndarray`, scalar, and other types.

- size:

  (list, tuple, or `torch.Size`, optional) Size of the sparse tensor. If
  not provided the size will be inferred as the minimum size big enough
  to hold all non-zero elements.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor.
  Default: if NULL, infers data type from `values`.

- device:

  (`torch.device`, optional) the desired device of returned tensor.
  Default: if NULL, uses the current device for the default tensor type
  (see `torch_set_default_tensor_type`). `device` will be the CPU for
  CPU tensor types and the current CUDA device for CUDA tensor types.

- requires_grad:

  (bool, optional) If autograd should record operations on the returned
  tensor. Default: `FALSE`.

## sparse_coo_tensor(indices, values, size=NULL, dtype=NULL, device=NULL, requires_grad=False) -\> Tensor

Constructs a sparse tensors in COO(rdinate) format with non-zero
elements at the given `indices` with the given `values`. A sparse tensor
can be `uncoalesced`, in that case, there are duplicate coordinates in
the indices, and the value at that index is the sum of all duplicate
value entries: `torch_sparse`\_.

## Examples

``` r
if (torch_is_installed()) {

i = torch_tensor(matrix(c(1, 2, 2, 3, 1, 3), ncol = 3, byrow = TRUE), dtype=torch_int64())
v = torch_tensor(c(3, 4, 5), dtype=torch_float32())
torch_sparse_coo_tensor(i, v)
torch_sparse_coo_tensor(i, v, c(2, 4))

# create empty sparse tensors
S = torch_sparse_coo_tensor(
  torch_empty(c(1, 0), dtype = torch_int64()), 
  torch_tensor(numeric(), dtype = torch_float32()), 
  c(1)
)
S = torch_sparse_coo_tensor(
  torch_empty(c(1, 0), dtype = torch_int64()), 
  torch_empty(c(0, 2)), 
  c(1, 2)
)
}
```
