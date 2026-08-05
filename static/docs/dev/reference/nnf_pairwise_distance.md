# Pairwise_distance

Computes the batchwise pairwise distance between vectors using the
p-norm.

## Usage

``` r
nnf_pairwise_distance(x1, x2, p = 2, eps = 1e-06, keepdim = FALSE)
```

## Arguments

- x1:

  (Tensor) First input.

- x2:

  (Tensor) Second input (of size matching x1).

- p:

  the norm degree. Default: 2

- eps:

  (float, optional) Small value to avoid division by zero. Default: 1e-8

- keepdim:

  Determines whether or not to keep the vector dimension. Default: False
