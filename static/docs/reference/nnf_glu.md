# Glu

The gated linear unit. Computes:

## Usage

``` r
nnf_glu(input, dim = -1)
```

## Arguments

- input:

  (Tensor) input tensor

- dim:

  (int) dimension on which to split the input. Default: -1

## Details

\$\$GLU(a, b) = a \otimes \sigma(b)\$\$

where `input` is split in half along `dim` to form `a` and `b`,
\\\sigma\\ is the sigmoid function and \\\otimes\\ is the element-wise
product between matrices.

See [Language Modeling with Gated Convolutional
Networks](https://arxiv.org/abs/1612.08083).
