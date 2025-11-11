# Cosine_similarity

Returns cosine similarity between x1 and x2, computed along dim.

## Usage

``` r
nnf_cosine_similarity(x1, x2, dim = 2, eps = 1e-08)
```

## Arguments

- x1:

  (Tensor) First input.

- x2:

  (Tensor) Second input (of size matching x1).

- dim:

  (int, optional) Dimension of vectors. Default: 2

- eps:

  (float, optional) Small value to avoid division by zero. Default: 1e-8

## Details

\$\$ \mbox{similarity} = \frac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert \_2
\cdot \Vert x_2 \Vert \_2, \epsilon)} \$\$
