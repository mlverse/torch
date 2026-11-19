# Extending Autograd

``` r
library(torch)
```

Adding operations to autograd requires implementing a new
`autograd_function` for each operation. Recall that
`autograd_functions`s are what `autograd` uses to compute the results
and gradients, and encode the operation history. Every new function
requires you to implement 2 methods:

- `forward()` - the code that performs the operation. It can take as
  many arguments as you want, with some of them being optional, if you
  specify the default values. All kinds of R objects are accepted here.
  Tensor arguments that track history (i.e., with `requires_grad=TRUE`)
  will be converted to ones that don’t track history before the call,
  and their use will be registered in the graph. Note that this logic
  won’t traverse lists or any other data structures and will only
  consider Tensor’s that are direct arguments to the call. You can
  return either a single Tensor output, or a list of `Tensor`s if there
  are multiple outputs. Also, please refer to the docs of
  `autograd_function` to find descriptions of useful methods that can be
  called only from `forward()`.

- `backward()` - gradient formula. It will be given as many Tensor
  arguments as there were outputs, with each of them representing
  gradient w.r.t. that output. It should return as many `Tensor`s as
  there were `Tensor's` that required gradients in `forward`, with each
  of them containing the gradient w.r.t. its corresponding input.

## Note

It’s the user’s responsibility to use the special functions in the
forward’s `ctx` properly in order to ensure that the new
`autograd_function` works properly with the autograd engine.

- `save_for_backward()` must be used when saving input or ouput of the
  forward to be used later in the backward.

- `mark_dirty()` must be used to mark any input that is modified inplace
  by the forward function.

- `mark_non_differentiable()` must be used to tell the engine if an
  output is not differentiable.

## Examples

Below you can find code for a linear function:

``` r
linear <- autograd_function(
  forward = function(ctx, input, weight, bias = NULL) {
    ctx$save_for_backward(input = input, weight = weight, bias = bias)
    output <- input$mm(weight$t())
    if (!is.null(bias))
      output <- output + bias$unsqueeze(0)$expand_as(output)
    
    output
  },
  backward = function(ctx, grad_output) {
    
    s <- ctx$saved_variables
    
    grads <- list(
      input = NULL,
      weight = NULL,
      bias = NULL
    )
    
    if (ctx$needs_input_grad$input)
      grads$input <- grad_output$mm(s$weight)
    
    if (ctx$needs_input_grad$weight)
      grads$weight <- grad_output$t()$mm(s$input)
    
    if (!is.null(s$bias) && ctx$needs_input_grad$bias)
      grads$bias <- grad_output$sum(dim = 0)
    
    grads
  }
)
```

Here, we give an additional example of a function that is parametrized
by non-Tensor arguments:

``` r
mul_constant <- autograd_function(
  forward = function(ctx, tensor, constant) {
    ctx$save_for_backward(constant = constant)
    tensor * constant
  },
  backward = function(ctx, grad_output) {
    v <- ctx$saved_variables
    list(
      tensor = grad_output * v$constant
    )
  }
)
```

``` r
x <- torch_tensor(1, requires_grad = TRUE)
o <- mul_constant(x, 2)
o$backward()
x$grad
#> torch_tensor
#>  2
#> [ CPUFloatType{1} ]
```
