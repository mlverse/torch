# Sparsemax

Applies the SparseMax activation.

## Usage

``` r
nnf_contrib_sparsemax(input, dim = -1)
```

## Arguments

- input:

  the input tensor

- dim:

  The dimension over which to apply the sparsemax function. (-1)

## Details

The SparseMax activation is described in ['From Softmax to Sparsemax: A
Sparse Model of Attention and Multi-Label
Classification'](https://arxiv.org/abs/1602.02068) The implementation is
based on
[aced125/sparsemax](https://github.com/aced125/sparsemax/tree/master/sparsemax)
