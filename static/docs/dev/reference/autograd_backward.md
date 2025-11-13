# Computes the sum of gradients of given tensors w.r.t. graph leaves.

The graph is differentiated using the chain rule. If any of tensors are
non-scalar (i.e. their data has more than one element) and require
gradient, then the Jacobian-vector product would be computed, in this
case the function additionally requires specifying `grad_tensors`. It
should be a sequence of matching length, that contains the “vector” in
the Jacobian-vector product, usually the gradient of the differentiated
function w.r.t. corresponding tensors (None is an acceptable value for
all tensors that don’t need gradient tensors).

## Usage

``` r
autograd_backward(
  tensors,
  grad_tensors = NULL,
  retain_graph = create_graph,
  create_graph = FALSE
)
```

## Arguments

- tensors:

  (list of Tensor) – Tensors of which the derivative will be computed.

- grad_tensors:

  (list of (Tensor or
  `NULL)) – The “vector” in the Jacobian-vector product, usually gradients w.r.t. each element of corresponding tensors. `NULL`values can be specified for scalar Tensors or ones that don’t require grad. If a`NULL\`
  value would be acceptable for all grad_tensors, then this argument is
  optional.

- retain_graph:

  (bool, optional) – If `FALSE`, the graph used to compute the grad will
  be freed. Note that in nearly all cases setting this option to `TRUE`
  is not needed and often can be worked around in a much more efficient
  way. Defaults to the value of `create_graph`.

- create_graph:

  (bool, optional) – If `TRUE`, graph of the derivative will be
  constructed, allowing to compute higher order derivative products.
  Defaults to `FALSE`.

## Details

This function accumulates gradients in the leaves - you might need to
zero them before calling it.

## Examples

``` r
if (torch_is_installed()) {
x <- torch_tensor(1, requires_grad = TRUE)
y <- 2 * x

a <- torch_tensor(1, requires_grad = TRUE)
b <- 3 * a

autograd_backward(list(y, b))
}
```
