# Call a (Potentially Unexported) Torch Function

This function allows calling a function prefixed with `torch_`,
including unexported functions which could have potentially valuable
uses but which do not yet have a user-friendly R wrapper function.
Therefore, this function should be used with extreme caution. Make sure
you understand what the function expects as input. It may be helpful to
read the `torch` source code for help with this, as well as the
documentation for the corresponding function in the Pytorch C++ API.
Generally for development and advanced use only.

## Usage

``` r
call_torch_function(name, ..., quiet = FALSE)
```

## Arguments

- name:

  Name of the function to call as a string. Should start with "torch\_"

- ...:

  A list of arguments to pass to the function. Argument splicing with
  `!!!` is supported.

- quiet:

  If TRUE, suppress warnings with valuable information about the dangers
  of this function.

## Value

The return value from calling the function `name` with arguments `...`

## Examples

``` r
if (torch_is_installed()) {
## many unexported functions do 'backward' calculations (e.g. derivatives)
## These could be used as a part of custom autograd functions for example.
x <- torch_randn(10, requires_grad = TRUE)
y <- torch_tanh(x)
## calculate backwards gradient using standard torch method
y$backward(torch_ones_like(x))
x$grad
## we can get the same result by calling the unexported `torch_tanh_backward()`
## function. The first argument is 1 to setup the Jacobian-vector product.
## see https://pytorch.org/blog/overview-of-pytorch-autograd-engine/ for details.
call_torch_function("torch_tanh_backward", 1, y)
all.equal(call_torch_function("torch_tanh_backward", 1, y, quiet = TRUE), x$grad)

}
#> Warning: Because this function allows access to unexported functions, please use with caution, and
#>             only if you are sure know what you are doing. Unexported functions will expect inputs that
#>             are more C++-like than R-like. For example, they will expect all indexes to be 0-based instead
#>             of 1-based. In addition unexported functions may be subject to removal from the API without
#>             warning. Set quiet = TRUE to silence this warning.
#> [1] TRUE
```
