# Embedding_bag

Computes sums, means or maxes of `bags` of embeddings, without
instantiating the intermediate embeddings.

## Usage

``` r
nnf_embedding_bag(
  input,
  weight,
  offsets = NULL,
  max_norm = NULL,
  norm_type = 2,
  scale_grad_by_freq = FALSE,
  mode = "mean",
  sparse = FALSE,
  per_sample_weights = NULL,
  include_last_offset = FALSE,
  padding_idx = NULL
)
```

## Arguments

- input:

  (LongTensor) Tensor containing bags of indices into the embedding
  matrix

- weight:

  (Tensor) The embedding matrix with number of rows equal to the maximum
  possible index + 1, and number of columns equal to the embedding size

- offsets:

  (LongTensor, optional) Only used when `input` is 1D. `offsets`
  determines the starting index position of each bag (sequence) in
  `input`.

- max_norm:

  (float, optional) If given, each embedding vector with norm larger
  than `max_norm` is renormalized to have norm `max_norm`. Note: this
  will modify `weight` in-place.

- norm_type:

  (float, optional) The `p` in the `p`-norm to compute for the
  `max_norm` option. Default `2`.

- scale_grad_by_freq:

  (boolean, optional) if given, this will scale gradients by the inverse
  of frequency of the words in the mini-batch. Default `FALSE`. Note:
  this option is not supported when `mode="max"`.

- mode:

  (string, optional) `"sum"`, `"mean"` or `"max"`. Specifies the way to
  reduce the bag. Default: 'mean'

- sparse:

  (bool, optional) if `TRUE`, gradient w.r.t. `weight` will be a sparse
  tensor. See Notes under `nn_embedding` for more details regarding
  sparse gradients. Note: this option is not supported when
  `mode="max"`.

- per_sample_weights:

  (Tensor, optional) a tensor of float / double weights, or NULL to
  indicate all weights should be taken to be 1. If specified,
  `per_sample_weights` must have exactly the same shape as input and is
  treated as having the same `offsets`, if those are not `NULL`.

- include_last_offset:

  (bool, optional) if `TRUE`, the size of offsets is equal to the number
  of bags + 1.

- padding_idx:

  (int, optional) If given, pads the output with the embedding vector at
  `padding_idx` (initialized to zeros) whenever it encounters the index.
