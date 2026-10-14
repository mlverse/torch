# Fractional_max_pool2d

Applies 2D fractional max pooling over an input signal composed of
several input planes.

## Usage

``` r
nnf_fractional_max_pool2d(
  input,
  kernel_size,
  output_size = NULL,
  output_ratio = NULL,
  return_indices = FALSE,
  random_samples = NULL
)
```

## Arguments

- input:

  the input tensor

- kernel_size:

  the size of the window to take a max over. Can be a single number
  \\k\\ (for a square kernel of \\k \* k\\) or a tuple `(kH, kW)`

- output_size:

  the target output size of the image of the form \\oH \* oW\\. Can be a
  tuple `(oH, oW)` or a single number \\oH\\ for a square image \\oH \*
  oH\\

- output_ratio:

  If one wants to have an output size as a ratio of the input size, this
  option can be given. This has to be a number or tuple in the range (0,
  1)

- return_indices:

  if `True`, will return the indices along with the outputs.

- random_samples:

  optional random samples.

## Details

Fractional MaxPooling is described in detail in the paper
`Fractional MaxPooling`\_ by Ben Graham

The max-pooling operation is applied in \\kH \* kW\\ regions by a
stochastic step size determined by the target output size. The number of
output features is equal to the number of input planes.
