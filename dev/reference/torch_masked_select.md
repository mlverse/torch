# Masked_select

Masked_select

## Usage

``` r
torch_masked_select(self, mask)
```

## Arguments

- self:

  (Tensor) the input tensor.

- mask:

  (BoolTensor) the tensor containing the binary mask to index with

## Note

The returned tensor does **not** use the same storage as the original
tensor

## masked_select(input, mask, out=NULL) -\> Tensor

Returns a new 1-D tensor which indexes the `input` tensor according to
the boolean mask `mask` which is a `BoolTensor`.

The shapes of the `mask` tensor and the `input` tensor don't need to
match, but they must be broadcastable .

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(3, 4))
x
mask = x$ge(0.5)
mask
torch_masked_select(x, mask)
}
#> torch_tensor
#> [ CPUFloatType{0} ]
```
