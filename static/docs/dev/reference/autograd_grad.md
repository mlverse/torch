# Computes and returns the sum of gradients of outputs w.r.t. the inputs.

`grad_outputs` should be a list of length matching output containing the
“vector” in Jacobian-vector product, usually the pre-computed gradients
w.r.t. each of the outputs. If an output doesn’t require_grad, then the
gradient can be None).

## Usage

``` r
autograd_grad(
  outputs,
  inputs,
  grad_outputs = NULL,
  retain_graph = create_graph,
  create_graph = FALSE,
  allow_unused = FALSE
)
```

## Arguments

- outputs:

  (sequence of Tensor) – outputs of the differentiated function.

- inputs:

  (sequence of Tensor) – Inputs w.r.t. which the gradient will be
  returned (and not accumulated into .grad).

- grad_outputs:

  (sequence of Tensor) – The “vector” in the Jacobian-vector product.
  Usually gradients w.r.t. each output. None values can be specified for
  scalar Tensors or ones that don’t require grad. If a None value would
  be acceptable for all `grad_tensors`, then this argument is optional.
  Default: None.

- retain_graph:

  (bool, optional) – If `FALSE`, the graph used to compute the grad will
  be freed. Note that in nearly all cases setting this option to `TRUE`
  is not needed and often can be worked around in a much more efficient
  way. Defaults to the value of `create_graph`.

- create_graph:

  (bool, optional) – If
  `TRUE, graph of the derivative will be constructed, allowing to compute higher order derivative products. Default: `FALSE\`.

- allow_unused:

  (bool, optional) – If `FALSE`, specifying inputs that were not used
  when computing outputs (and therefore their grad is always zero) is an
  error. Defaults to `FALSE`

## Details

If only_inputs is `TRUE`, the function will only return a list of
gradients w.r.t the specified inputs. If it’s `FALSE`, then gradient
w.r.t. all remaining leaves will still be computed, and will be
accumulated into their `.grad` attribute.

## Examples

``` r
if (torch_is_installed()) {
w <- torch_tensor(0.5, requires_grad = TRUE)
b <- torch_tensor(0.9, requires_grad = TRUE)
x <- torch_tensor(runif(100))
y <- 2 * x + 1
loss <- (y - (w * x + b))^2
loss <- loss$mean()

o <- autograd_grad(loss, list(w, b))
o
}
#> [[1]]
#> torch_tensor
#> -1.0210
#> [ CPUFloatType{1} ]
#> 
#> [[2]]
#> torch_tensor
#> -1.6561
#> [ CPUFloatType{1} ]
#> 
```
