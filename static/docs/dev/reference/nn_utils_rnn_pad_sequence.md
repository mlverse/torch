# Pad a list of variable length Tensors with `padding_value`

`pad_sequence` stacks a list of Tensors along a new dimension, and pads
them to equal length. For example, if the input is list of sequences
with size `L x *` and if batch_first is False, and `T x B x *`
otherwise.

## Usage

``` r
nn_utils_rnn_pad_sequence(sequences, batch_first = FALSE, padding_value = 0)
```

## Arguments

- sequences:

  `(list[Tensor])`: list of variable length sequences.

- batch_first:

  (bool, optional): output will be in `B x T x *` if `TRUE`, or in
  `T x B x *` otherwise

- padding_value:

  (float, optional): value for padded elements. Default: 0.

## Value

Tensor of size `T x B x *` if `batch_first` is `FALSE`. Tensor of size
`B x T x *` otherwise

## Details

`B` is batch size. It is equal to the number of elements in `sequences`.
`T` is length of the longest sequence. `L` is length of the sequence.
`*` is any number of trailing dimensions, including none.

## Note

This function returns a Tensor of size `T x B x *` or `B x T x *` where
`T` is the length of the longest sequence. This function assumes
trailing dimensions and type of all the Tensors in sequences are same.

## Examples

``` r
if (torch_is_installed()) {
a <- torch_ones(25, 300)
b <- torch_ones(22, 300)
c <- torch_ones(15, 300)
nn_utils_rnn_pad_sequence(list(a, b, c))$size()
}
#> [1]  25   3 300
```
