# Unfold

Extracts sliding local blocks from an batched input tensor.

## Usage

``` r
nnf_unfold(input, kernel_size, dilation = 1, padding = 0, stride = 1)
```

## Arguments

- input:

  the input tensor

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

More than one element of the unfolded tensor may refer to a single
memory location. As a result, in-place operations (especially ones that
are vectorized) may result in incorrect behavior. If you need to write
to the tensor, please clone it first.
