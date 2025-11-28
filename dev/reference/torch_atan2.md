# Atan2

Atan2

## Usage

``` r
torch_atan2(self, other)
```

## Arguments

- self:

  (Tensor) the first input tensor

- other:

  (Tensor) the second input tensor

## atan2(input, other, out=NULL) -\> Tensor

Element-wise arctangent of \\\mbox{input}\_{i} / \mbox{other}\_{i}\\
with consideration of the quadrant. Returns a new tensor with the signed
angles in radians between vector \\(\mbox{other}\_{i},
\mbox{input}\_{i})\\ and vector \\(1, 0)\\. (Note that
\\\mbox{other}\_{i}\\, the second parameter, is the x-coordinate, while
\\\mbox{input}\_{i}\\, the first parameter, is the y-coordinate.)

The shapes of `input` and `other` must be broadcastable .

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_atan2(a, torch_randn(c(4)))
}
#> torch_tensor
#> -1.2455
#>  0.2797
#>  1.4308
#> -0.2293
#> [ CPUFloatType{4} ]
```
