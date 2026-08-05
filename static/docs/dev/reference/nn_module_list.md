# Holds submodules in a list.

nn_module_list can be indexed like a regular R list, but modules it
contains are properly registered, and will be visible by all `nn_module`
methods.

## Usage

``` r
nn_module_list(modules = list())
```

## Arguments

- modules:

  a list of modules to add

## See also

[`nn_module_dict()`](https://torch.mlverse.org/docs/dev/reference/nn_module_dict.md)

## Examples

``` r
if (torch_is_installed()) {

my_module <- nn_module(
  initialize = function() {
    self$linears <- nn_module_list(lapply(1:10, function(x) nn_linear(10, 10)))
  },
  forward = function(x) {
    for (i in 1:length(self$linears)) {
      x <- self$linears[[i]](x)
    }
    x
  }
)
}
```
