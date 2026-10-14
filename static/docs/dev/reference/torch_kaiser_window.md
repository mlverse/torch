# Kaiser_window

Kaiser_window

## Usage

``` r
torch_kaiser_window(
  window_length,
  periodic,
  beta,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = NULL
)
```

## Arguments

- window_length:

  (int) length of the window.

- periodic:

  (bool, optional) If TRUE, returns a periodic window suitable for use
  in spectral analysis. If FALSE, returns a symmetric window suitable
  for use in filter design.

- beta:

  (float, optional) shape parameter for the window.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor.
  Default: if `NULL`, uses a global default (see
  `torch_set_default_tensor_type`). If `dtype` is not given, infer the
  data type from the other input arguments. If any of `start`, `end`, or
  `stop` are floating-point, the `dtype` is inferred to be the default
  dtype, see `~torch.get_default_dtype`. Otherwise, the `dtype` is
  inferred to be `torch.int64`.

- layout:

  (`torch.layout`, optional) the desired layout of returned Tensor.
  Default: `torch_strided`.

- device:

  (`torch.device`, optional) the desired device of returned tensor.
  Default: if `NULL`, uses the current device for the default tensor
  type (see `torch_set_default_tensor_type`). `device` will be the CPU
  for CPU tensor types and the current CUDA device for CUDA tensor
  types.

- requires_grad:

  (bool, optional) If autograd should record operations on the returned
  tensor. Default: `FALSE`.

## Note

If `window_length` is one, then the returned window is a single element
tensor containing a one.

## kaiser_window(window_length, periodic=TRUE, beta=12.0, \*, dtype=None, layout=torch.strided, device=None, requires_grad=FALSE) -\> Tensor

Computes the Kaiser window with window length `window_length` and shape
parameter `beta`.

Let I_0 be the zeroth order modified Bessel function of the first kind
(see
[`torch_i0()`](https://torch.mlverse.org/docs/dev/reference/torch_i0.md))
and `N = L - 1` if `periodic` is FALSE and `L` if `periodic` is TRUE,
where `L` is the `window_length`. This function computes:

\$\$ out_i = I_0 \left( \beta \sqrt{1 - \left( {\frac{i - N/2}{N/2}}
\right) ^2 } \right) / I_0( \beta ) \$\$

Calling `torch_kaiser_window(L, B, periodic=TRUE)` is equivalent to
calling `torch_kaiser_window(L + 1, B, periodic=FALSE)[:-1])`. The
`periodic` argument is intended as a helpful shorthand to produce a
periodic window as input to functions like
[`torch_stft()`](https://torch.mlverse.org/docs/dev/reference/torch_stft.md).
