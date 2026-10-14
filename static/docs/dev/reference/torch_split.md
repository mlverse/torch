# Split

Splits the tensor into chunks. Each chunk is a view of the original
tensor.

## Usage

``` r
torch_split(self, split_size, dim = 1L)
```

## Arguments

- self:

  (Tensor) tensor to split.

- split_size:

  (int) size of a single chunk or list of sizes for each chunk

- dim:

  (int) dimension along which to split the tensor.

## Details

If `split_size` is an integer type, then `tensor` will be split into
equally sized chunks (if possible). Last chunk will be smaller if the
tensor size along the given dimension `dim` is not divisible by
`split_size`.

If `split_size` is a list, then `tensor` will be split into
`length(split_size)` chunks with sizes in `dim` according to
`split_size_or_sections`.
