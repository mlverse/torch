# Fractional_max_pool3d

Applies 3D fractional max pooling over an input signal composed of
several input planes.

## Usage

``` r
nnf_fractional_max_pool3d(
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
  \\k\\ (for a square kernel of \\k \* k \* k\\) or a tuple
  `(kT, kH, kW)`

- output_size:

  the target output size of the form \\oT \* oH \* oW\\. Can be a tuple
  `(oT, oH, oW)` or a single number \\oH\\ for a cubic output \\oH \* oH
  \* oH\\

- output_ratio:

  If one wants to have an output size as a ratio of the input size, this
  option can be given. This has to be a number or tuple in the range (0,
  1)

- return_indices:

  if `True`, will return the indices along with the outputs.

- random_samples:

  undocumented argument.

## Details

Fractional MaxPooling is described in detail in the paper
`Fractional MaxPooling`\_ by Ben Graham

The max-pooling operation is applied in \\kT \* kH \* kW\\ regions by a
stochastic step size determined by the target output size. The number of
output features is equal to the number of input planes.
