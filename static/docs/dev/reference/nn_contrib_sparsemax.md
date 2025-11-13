# Sparsemax activation

Sparsemax activation module.

## Usage

``` r
nn_contrib_sparsemax(dim = -1)
```

## Arguments

- dim:

  The dimension over which to apply the sparsemax function. (-1)

## Details

The SparseMax activation is described in ['From Softmax to Sparsemax: A
Sparse Model of Attention and Multi-Label
Classification'](https://arxiv.org/abs/1602.02068) The implementation is
based on
[aced125/sparsemax](https://github.com/aced125/sparsemax/tree/master/sparsemax)
