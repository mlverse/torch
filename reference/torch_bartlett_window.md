# Bartlett_window

Bartlett_window

## Usage

``` r
torch_bartlett_window(
  window_length,
  periodic = TRUE,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- window_length:

  (int) the size of returned window

- periodic:

  (bool, optional) If TRUE, returns a window to be used as periodic
  function. If False, return a symmetric window.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor.
  Default: if `NULL`, uses a global default (see
  `torch_set_default_tensor_type`). Only floating point types are
  supported.

- layout:

  (`torch.layout`, optional) the desired layout of returned window
  tensor. Only `torch_strided` (dense layout) is supported.

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

    If `window_length` \eqn{=1}, the returned window contains a single value 1.

## bartlett_window(window_length, periodic=TRUE, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Bartlett window function.

\$\$ w\[n\] = 1 - \left\| \frac{2n}{N-1} - 1 \right\| = \left\\
\begin{array}{ll} \frac{2n}{N - 1} & \mbox{if } 0 \leq n \leq \frac{N -
1}{2} \\ 2 - \frac{2n}{N - 1} & \mbox{if } \frac{N - 1}{2} \< n \< N \\
\end{array} \right. , \$\$ where \\N\\ is the full window size.

The input `window_length` is a positive integer controlling the returned
window size. `periodic` flag determines whether the returned window
trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like `torch_stft`.
Therefore, if `periodic` is true, the \\N\\ in above formula is in fact
\\\mbox{window\\length} + 1\\. Also, we always have
`torch_bartlett_window(L, periodic=TRUE)` equal to
`torch_bartlett_window(L + 1, periodic=False)[:-1])`.
