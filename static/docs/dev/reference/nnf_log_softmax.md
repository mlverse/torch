# Log_softmax

Applies a softmax followed by a logarithm.

## Usage

``` r
nnf_log_softmax(input, dim = NULL, dtype = NULL)
```

## Arguments

- input:

  (Tensor) input

- dim:

  (int) A dimension along which log_softmax will be computed.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor. If
  specified, the input tensor is casted to `dtype` before the operation
  is performed. This is useful for preventing data type overflows.
  Default: `NULL`.

## Details

While mathematically equivalent to log(softmax(x)), doing these two
operations separately is slower, and numerically unstable. This function
uses an alternative formulation to compute the output and gradient
correctly.
