# Temporarily modify gradient recording.

Temporarily modify gradient recording.

## Usage

``` r
with_no_grad(code)

local_no_grad(.env = parent.frame())
```

## Arguments

- code:

  code to be executed with no gradient recording.

- .env:

  The environment to use for scoping.

## Functions

- `local_no_grad()`: Disable autograd until it goes out of scope

## Examples

``` r
if (torch_is_installed()) {
x <- torch_tensor(runif(5), requires_grad = TRUE)
with_no_grad({
  x$sub_(torch_tensor(as.numeric(1:5)))
})
x
x$grad
}
#> torch_tensor
#> [ Tensor (undefined) ]
```
