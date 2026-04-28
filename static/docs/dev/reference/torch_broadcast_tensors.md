# Broadcast_tensors

Broadcast_tensors

## Usage

``` r
torch_broadcast_tensors(tensors)
```

## Arguments

- tensors:

  a list containing any number of tensors of the same type

## broadcast_tensors(tensors) -\> List of Tensors

Broadcasts the given tensors according to broadcasting-semantics.

## Examples

``` r
if (torch_is_installed()) {

x = torch_arange(0, 3)$view(c(1, 4))
y = torch_arange(0, 2)$view(c(3, 1))
out = torch_broadcast_tensors(list(x, y))
out[[1]]
}
#> torch_tensor
#>  0  1  2  3
#>  0  1  2  3
#>  0  1  2  3
#> [ CPUFloatType{3,4} ]
```
