# Stack

Stack

## Usage

``` r
torch_stack(tensors, dim = 1L)
```

## Arguments

- tensors:

  (sequence of Tensors) sequence of tensors to concatenate

- dim:

  (int) dimension to insert. Has to be between 0 and the number of
  dimensions of concatenated tensors (inclusive)

## stack(tensors, dim=0, out=NULL) -\> Tensor

Concatenates sequence of tensors along a new dimension.

All tensors need to be of the same size.
