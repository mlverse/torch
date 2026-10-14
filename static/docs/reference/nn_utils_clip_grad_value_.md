# Clips gradient of an iterable of parameters at specified value.

Gradients are modified in-place.

## Usage

``` r
nn_utils_clip_grad_value_(parameters, clip_value)
```

## Arguments

- parameters:

  (Iterable(Tensor) or Tensor): an iterable of Tensors or a single
  Tensor that will have gradients normalized

- clip_value:

  (float or int): maximum allowed value of the gradients.

## Details

The gradients are clipped in the range \\\left\[\mbox{-clip\\value},
\mbox{clip\\value}\right\]\\
