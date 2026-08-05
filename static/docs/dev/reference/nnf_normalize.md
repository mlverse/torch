# Normalize

Performs \\L_p\\ normalization of inputs over specified dimension.

## Usage

``` r
nnf_normalize(input, p = 2, dim = 2, eps = 1e-12, out = NULL)
```

## Arguments

- input:

  input tensor of any shape

- p:

  (float) the exponent value in the norm formulation. Default: 2

- dim:

  (int) the dimension to reduce. Default: 1

- eps:

  (float) small value to avoid division by zero. Default: 1e-12

- out:

  (Tensor, optional) the output tensor. If `out` is used, this operation
  won't be differentiable.

## Details

For a tensor `input` of sizes \\(n_0, ..., n\_{dim}, ..., n_k)\\, each
\\n\_{dim}\\ -element vector \\v\\ along dimension `dim` is transformed
as

\$\$ v = \frac{v}{\max(\Vert v \Vert_p, \epsilon)}. \$\$

With the default arguments it uses the Euclidean norm over vectors along
dimension \\1\\ for normalization.
