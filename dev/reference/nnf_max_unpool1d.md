# Max_unpool1d

Computes a partial inverse of `MaxPool1d`.

## Usage

``` r
nnf_max_unpool1d(
  input,
  indices,
  kernel_size,
  stride = NULL,
  padding = 0,
  output_size = NULL
)
```

## Arguments

- input:

  the input Tensor to invert

- indices:

  the indices given out by max pool

- kernel_size:

  Size of the max pooling window.

- stride:

  Stride of the max pooling window. It is set to kernel_size by default.

- padding:

  Padding that was added to the input

- output_size:

  the targeted output size
