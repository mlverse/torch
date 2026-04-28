# Cdist

Cdist

## Usage

``` r
torch_cdist(x1, x2, p = 2L, compute_mode = NULL)
```

## Arguments

- x1:

  (Tensor) input tensor of shape \\B \times P \times M\\.

- x2:

  (Tensor) input tensor of shape \\B \times R \times M\\.

- p:

  NA p value for the p-norm distance to calculate between each vector
  pair \\\in \[0, \infty\]\\.

- compute_mode:

  NA 'use_mm_for_euclid_dist_if_necessary' - will use matrix
  multiplication approach to calculate euclidean distance (p = 2) if P
  \> 25 or R \> 25 'use_mm_for_euclid_dist' - will always use matrix
  multiplication approach to calculate euclidean distance (p = 2)
  'donot_use_mm_for_euclid_dist' - will never use matrix multiplication
  approach to calculate euclidean distance (p = 2) Default:
  use_mm_for_euclid_dist_if_necessary.

## TEST

Computes batched the p-norm distance between each pair of the two
collections of row vectors.
