# Creates a tensor from a buffer of memory

It creates a tensor without taking ownership of the memory it points to.
You must call `clone` if you want to copy the memory over a new tensor.

## Usage

``` r
torch_tensor_from_buffer(buffer, shape, dtype = "float")

buffer_from_torch_tensor(tensor)
```

## Arguments

- buffer:

  An R atomic object containing the data in a contiguous array.

- shape:

  The shape of the resulting tensor.

- dtype:

  A torch data type for the tresulting tensor.

- tensor:

  Tensor object that will be converted into a buffer.

## Functions

- `buffer_from_torch_tensor()`: Creates a raw vector containing the
  tensor data. Causes a data copy.
