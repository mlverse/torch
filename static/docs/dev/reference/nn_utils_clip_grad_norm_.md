# Clips gradient norm of an iterable of parameters.

The norm is computed over all gradients together, as if they were
concatenated into a single vector. Gradients are modified in-place.

## Usage

``` r
nn_utils_clip_grad_norm_(parameters, max_norm, norm_type = 2)
```

## Arguments

- parameters:

  (IterableTensor or Tensor): an iterable of Tensors or a single Tensor
  that will have gradients normalized

- max_norm:

  (float or int): max norm of the gradients

- norm_type:

  (float or int): type of the used p-norm. Can be `Inf` for infinity
  norm.

## Value

Total norm of the parameters (viewed as a single vector).
