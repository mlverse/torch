# Clone a torch module.

Clones a module.

## Usage

``` r
clone_module(module, deep = FALSE, ..., replace_values = TRUE)
```

## Arguments

- module:

  ([`nn_module`](https://torch.mlverse.org/docs/dev/reference/nn_module.md))  
  The module to clone

- deep:

  (`logical(1)`)  
  Whether to create a deep clone.

- ...:

  (any)  
  Additional parameters, currently unused.

- replace_values:

  (`logical(1)`)  
  Whether to replace parameters and buffers with the cloned values.

## Examples

``` r
if (torch_is_installed()) {
clone_module(nn_linear(1, 1), deep = TRUE)
# is the same as
nn_linear(1, 1)$clone(deep = TRUE)
}
#> An `nn_module` containing 2 parameters.
#> 
#> ── Parameters ──────────────────────────────────────────────────────────────────
#> • weight: Float [1:1, 1:1]
#> • bias: Float [1:1]
```
