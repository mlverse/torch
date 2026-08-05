# Container that allows named values

Container that allows named values

## Usage

``` r
nn_module_dict(dict)
```

## Arguments

- dict:

  A named list of submodules that will be saved in that module.

## See also

[`nn_module_list()`](https://torch.mlverse.org/docs/dev/reference/nn_module_list.md)

## Examples

``` r
if (torch_is_installed()) {
nn_module <- nn_module(
  initialize = function() {
    self$dict <- nn_module_dict(list(
      l1 = nn_linear(10, 20),
      l2 = nn_linear(20, 10)
    ))
  },
  forward = function(x) {
    x <- self$dict$l1(x)
    self$dict$l2(x)
  }
)
}
```
