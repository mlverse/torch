# Lerp

Lerp

## Usage

``` r
torch_lerp(self, end, weight)
```

## Arguments

- self:

  (Tensor) the tensor with the starting points

- end:

  (Tensor) the tensor with the ending points

- weight:

  (float or tensor) the weight for the interpolation formula

## lerp(input, end, weight, out=NULL)

Does a linear interpolation of two tensors `start` (given by `input`)
and `end` based on a scalar or tensor `weight` and returns the resulting
`out` tensor.

\$\$ \mbox{out}\_i = \mbox{start}\_i + \mbox{weight}\_i \times
(\mbox{end}\_i - \mbox{start}\_i) \$\$ The shapes of `start` and `end`
must be broadcastable . If `weight` is a tensor, then the shapes of
`weight`, `start`, and `end` must be broadcastable .

## Examples

``` r
if (torch_is_installed()) {

start = torch_arange(1, 4)
end = torch_empty(4)$fill_(10)
start
end
torch_lerp(start, end, 0.5)
torch_lerp(start, end, torch_full_like(start, 0.5))
}
#> torch_tensor
#>  5.5000
#>  6.0000
#>  6.5000
#>  7.0000
#> [ CPUFloatType{4} ]
```
