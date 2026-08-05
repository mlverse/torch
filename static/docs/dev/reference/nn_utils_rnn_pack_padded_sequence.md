# Packs a Tensor containing padded sequences of variable length.

`input` can be of size `T x B x *` where `T` is the length of the
longest sequence (equal to `lengths[1]`), `B` is the batch size, and `*`
is any number of dimensions (including 0). If `batch_first` is `TRUE`,
`B x T x *` `input` is expected.

## Usage

``` r
nn_utils_rnn_pack_padded_sequence(
  input,
  lengths,
  batch_first = FALSE,
  enforce_sorted = TRUE
)
```

## Arguments

- input:

  (Tensor): padded batch of variable length sequences.

- lengths:

  (Tensor): list of sequences lengths of each batch element.

- batch_first:

  (bool, optional): if `TRUE`, the input is expected in `B x T x *`
  format.

- enforce_sorted:

  (bool, optional): if `TRUE`, the input is expected to contain
  sequences sorted by length in a decreasing order. If `FALSE`, the
  input will get sorted unconditionally. Default: `TRUE`.

## Value

a `PackedSequence` object

## Details

For unsorted sequences, use `enforce_sorted = FALSE`. If
`enforce_sorted` is `TRUE`, the sequences should be sorted by length in
a decreasing order, i.e. `input[,1]` should be the longest sequence, and
`input[,B]` the shortest one. `enforce_sorted = TRUE` is only necessary
for ONNX export.

## Note

This function accepts any input that has at least two dimensions. You
can apply it to pack the labels, and use the output of the RNN with them
to compute the loss directly. A Tensor can be retrieved from a
`PackedSequence` object by accessing its `.data` attribute.
