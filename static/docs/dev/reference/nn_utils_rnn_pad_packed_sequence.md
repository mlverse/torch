# Pads a packed batch of variable length sequences.

It is an inverse operation to
[`nn_utils_rnn_pack_padded_sequence()`](https://torch.mlverse.org/docs/dev/reference/nn_utils_rnn_pack_padded_sequence.md).

## Usage

``` r
nn_utils_rnn_pad_packed_sequence(
  sequence,
  batch_first = FALSE,
  padding_value = 0,
  total_length = NULL
)
```

## Arguments

- sequence:

  (PackedSequence): batch to pad

- batch_first:

  (bool, optional): if `True`, the output will be in “B x T x \*\`
  format.

- padding_value:

  (float, optional): values for padded elements.

- total_length:

  (int, optional): if not `NULL`, the output will be padded to have
  length `total_length`. This method will throw `ValueError` if
  `total_length` is less than the max sequence length in `sequence`.

## Value

Tuple of Tensor containing the padded sequence, and a Tensor containing
the list of lengths of each sequence in the batch. Batch elements will
be re-ordered as they were ordered originally when the batch was passed
to
[`nn_utils_rnn_pack_padded_sequence()`](https://torch.mlverse.org/docs/dev/reference/nn_utils_rnn_pack_padded_sequence.md)
or
[`nn_utils_rnn_pack_sequence()`](https://torch.mlverse.org/docs/dev/reference/nn_utils_rnn_pack_sequence.md).

## Details

The returned Tensor's data will be of size `T x B x *`, where `T` is the
length of the longest sequence and `B` is the batch size. If
`batch_first` is `TRUE`, the data will be transposed into `B x T x *`
format.

## Note

`total_length` is useful to implement the
`pack sequence -> recurrent network -> unpack sequence` pattern in a
`nn_module` wrapped in `~torch.nn.DataParallel`.

## Examples

``` r
if (torch_is_installed()) {
seq <- torch_tensor(rbind(c(1, 2, 0), c(3, 0, 0), c(4, 5, 6)))
lens <- c(2, 1, 3)
packed <- nn_utils_rnn_pack_padded_sequence(seq, lens,
  batch_first = TRUE,
  enforce_sorted = FALSE
)
packed
nn_utils_rnn_pad_packed_sequence(packed, batch_first = TRUE)
}
#> [[1]]
#> torch_tensor
#>  1  2  0
#>  3  0  0
#>  4  5  6
#> [ CPUFloatType{3,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  2
#>  1
#>  3
#> [ CPULongType{3} ]
#> 
```
