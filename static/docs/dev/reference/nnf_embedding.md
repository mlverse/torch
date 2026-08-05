# Embedding

A simple lookup table that looks up embeddings in a fixed dictionary and
size.

## Usage

``` r
nnf_embedding(
  input,
  weight,
  padding_idx = NULL,
  max_norm = NULL,
  norm_type = 2,
  scale_grad_by_freq = FALSE,
  sparse = FALSE
)
```

## Arguments

- input:

  (LongTensor) Tensor containing indices into the embedding matrix

- weight:

  (Tensor) The embedding matrix with number of rows equal to the maximum
  possible index + 1, and number of columns equal to the embedding size

- padding_idx:

  (int, optional) If given, pads the output with the embedding vector at
  `padding_idx` (initialized to zeros) whenever it encounters the index.

- max_norm:

  (float, optional) If given, each embedding vector with norm larger
  than `max_norm` is renormalized to have norm `max_norm`. Note: this
  will modify `weight` in-place.

- norm_type:

  (float, optional) The p of the p-norm to compute for the `max_norm`
  option. Default `2`.

- scale_grad_by_freq:

  (boolean, optional) If given, this will scale gradients by the inverse
  of frequency of the words in the mini-batch. Default `FALSE`.

- sparse:

  (bool, optional) If `TRUE`, gradient w.r.t. `weight` will be a sparse
  tensor. See Notes under `nn_embedding` for more details regarding
  sparse gradients.

## Details

This module is often used to retrieve word embeddings using indices. The
input to the module is a list of indices, and the embedding matrix, and
the output is the corresponding word embeddings.
