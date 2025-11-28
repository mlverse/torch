# Dist

Dist

## Usage

``` r
torch_dist(self, other, p = 2L)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the Right-hand-side input tensor

- p:

  (float, optional) the norm to be computed

## dist(input, other, p=2) -\> Tensor

Returns the p-norm of (`input` - `other`)

The shapes of `input` and `other` must be broadcastable .

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(4))
x
y = torch_randn(c(4))
y
torch_dist(x, y, 3.5)
torch_dist(x, y, 3)
torch_dist(x, y, 0)
torch_dist(x, y, 1)
}
#> torch_tensor
#> 2.6706
#> [ CPUFloatType{} ]
```
