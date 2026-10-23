# Chunk

Chunk

## Usage

``` r
torch_chunk(self, chunks, dim = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to split

- chunks:

  (int) number of chunks to return

- dim:

  (int) dimension along which to split the tensor

## chunk(input, chunks, dim=0) -\> List of Tensors

Splits a tensor into a specific number of chunks. Each chunk is a view
of the input tensor.

Last chunk will be smaller if the tensor size along the given dimension
`dim` is not divisible by `chunks`.
