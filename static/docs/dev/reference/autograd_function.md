# Records operation history and defines formulas for differentiating ops.

Every operation performed on Tensor's creates a new function object,
that performs the computation, and records that it happened. The history
is retained in the form of a DAG of functions, with edges denoting data
dependencies (input \<- output). Then, when backward is called, the
graph is processed in the topological ordering, by calling `backward()`
methods of each Function object, and passing returned gradients on to
next Function's.

## Usage

``` r
autograd_function(forward, backward)
```

## Arguments

- forward:

  Performs the operation. It must accept a context `ctx` as the first
  argument, followed by any number of arguments (tensors or other
  types). The context can be used to store tensors that can be then
  retrieved during the backward pass. See
  [AutogradContext](https://torch.mlverse.org/docs/dev/reference/AutogradContext.md)
  for more information about context methods.

- backward:

  Defines a formula for differentiating the operation. It must accept a
  context `ctx` as the first argument, followed by as many outputs ad
  `forward()` returned (as a
  [`list()`](https://rdrr.io/r/base/list.html)). The names of the
  arguments don't matter and they are passed in the order in which they
  were returned by `forward()`. The function should return a named list,
  where each argument is the gradient w.r.t the given output, and each
  element in the returned list should be the gradient w.r.t. the
  corresponding input. The context can be used to retrieve tensors saved
  during the forward pass. It also has an attribute
  `ctx$needs_input_grad` as a named list of booleans representing
  whether each input needs gradient. E.g., `backward()` will have
  `ctx$needs_input_grad$input = TRUE` if the `input` argument to
  `forward()` needs gradient computated w.r.t. the output. See
  [AutogradContext](https://torch.mlverse.org/docs/dev/reference/AutogradContext.md)
  for more information about context methods.

## Examples

``` r
if (torch_is_installed()) {

exp2 <- autograd_function(
  forward = function(ctx, i) {
    result <- i$exp()
    ctx$save_for_backward(result = result)
    result
  },
  backward = function(ctx, grad_output) {
    list(i = grad_output * ctx$saved_variable$result)
  }
)
}
```
