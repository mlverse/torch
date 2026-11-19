# Enable grad

Context-manager that enables gradient calculation. Enables gradient
calculation, if it has been disabled via
[with_no_grad](https://torch.mlverse.org/docs/dev/reference/with_no_grad.md).

## Usage

``` r
with_enable_grad(code)

local_enable_grad(.env = parent.frame())
```

## Arguments

- code:

  code to be executed with gradient recording.

- .env:

  The environment to use for scoping.

## Details

This context manager is thread local; it will not affect computation in
other threads.

## Functions

- `local_enable_grad()`: Locally enable gradient computations.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_tensor(1, requires_grad = TRUE)
with_no_grad({
  with_enable_grad({
    y <- x * 2
  })
})
y$backward()
x$grad
}
#> torch_tensor
#>  2
#> [ CPUFloatType{1} ]
```
