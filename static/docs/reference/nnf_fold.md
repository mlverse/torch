# Fold

Combines an array of sliding local blocks into a large containing
tensor.

## Usage

``` r
nnf_fold(
  input,
  output_size,
  kernel_size,
  dilation = 1,
  padding = 0,
  stride = 1
)
```

## Arguments

- input:

  the input tensor

- output_size:

  the shape of the spatial dimensions of the output (i.e.,
  `output$sizes()[-c(1,2)]`)

- kernel_size:

  the size of the sliding blocks

- dilation:

  a parameter that controls the stride of elements within the
  neighborhood. Default: 1

- padding:

  implicit zero padding to be added on both sides of input. Default: 0

- stride:

  the stride of the sliding blocks in the input spatial dimensions.
  Default: 1

## Warning

Currently, only 4-D output tensors (batched image-like tensors) are
supported.
