# Packs a list of variable length Tensors

`sequences` should be a list of Tensors of size `L x *`, where `L` is
the length of a sequence and `*` is any number of trailing dimensions,
including zero.

## Usage

``` r
nn_utils_rnn_pack_sequence(sequences, enforce_sorted = TRUE)
```

## Arguments

- sequences:

  `(list[Tensor])`: A list of sequences of decreasing length.

- enforce_sorted:

  (bool, optional): if `TRUE`, checks that the input contains sequences
  sorted by length in a decreasing order. If `FALSE`, this condition is
  not checked. Default: `TRUE`.

## Value

a `PackedSequence` object

## Details

For unsorted sequences, use `enforce_sorted = FALSE`. If
`enforce_sorted` is `TRUE`, the sequences should be sorted in the order
of decreasing length. `enforce_sorted = TRUE` is only necessary for ONNX
export.

## Examples

``` r
if (torch_is_installed()) {
x <- torch_tensor(c(1, 2, 3), dtype = torch_long())
y <- torch_tensor(c(4, 5), dtype = torch_long())
z <- torch_tensor(c(6), dtype = torch_long())

p <- nn_utils_rnn_pack_sequence(list(x, y, z))
}
```
