# Prune top layer(s) of a network

Prune `head_size` last layers of a nn_module in order to replace them by
your own head, or in order to use the pruned module as a sequential
embedding module.

## Usage

``` r
nn_prune_head(x, head_size)
```

## Arguments

- x:

  nn_network to prune

- head_size:

  number of nn_layers to prune

## Value

a nn_sequential network with the top nn_layer removed

## Examples

``` r
if (torch_is_installed()) {
if (torch_is_installed()) {
x <- nn_sequential(
  nn_relu(),
  nn_tanh(),
  nn_relu6(),
  nn_relu(),
  nn_linear(2,10),
  nn_batch_norm1d(10),
  nn_tanh(),
  nn_linear(10,3)
)
prune <- nn_prune_head(x, 3)
prune
}
}
#> An `nn_module` containing 30 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • 0: <nn_relu> #0 parameters
#> • 1: <nn_tanh> #0 parameters
#> • 2: <nn_relu6> #0 parameters
#> • 3: <nn_relu> #0 parameters
#> • 4: <nn_linear> #30 parameters
```
