# Ctc_loss

The Connectionist Temporal Classification loss.

## Usage

``` r
nnf_ctc_loss(
  log_probs,
  targets,
  input_lengths,
  target_lengths,
  blank = 0,
  reduction = c("mean", "sum", "none"),
  zero_infinity = FALSE
)
```

## Arguments

- log_probs:

  \\(T, N, C)\\ where C = number of characters in alphabet including
  blank, T = input length, and N = batch size. The logarithmized
  probabilities of the outputs (e.g. obtained with
  [nnf_log_softmax](https://torch.mlverse.org/docs/dev/reference/nnf_log_softmax.md)).

- targets:

  \\(N, S)\\ or `(sum(target_lengths))`. Targets cannot be blank. In the
  second form, the targets are assumed to be concatenated.

- input_lengths:

  \\(N)\\. Lengths of the inputs (must each be \\\leq T\\)

- target_lengths:

  \\(N)\\. Lengths of the targets

- blank:

  (int, optional) Blank label. Default \\0\\.

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'

- zero_infinity:

  (bool, optional) Whether to zero infinite losses and the associated
  gradients. Default: `FALSE` Infinite losses mainly occur when the
  inputs are too short to be aligned to the targets.
