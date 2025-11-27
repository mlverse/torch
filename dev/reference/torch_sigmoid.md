# Sigmoid

Sigmoid

## Usage

``` r
torch_sigmoid(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## sigmoid(input, out=NULL) -\> Tensor

Returns a new tensor with the sigmoid of the elements of `input`.

\$\$ \mbox{out}\_{i} = \frac{1}{1 + e^{-\mbox{input}\_{i}}} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_sigmoid(a)
}
#> torch_tensor
#>  0.5703
#>  0.4710
#>  0.4976
#>  0.7084
#> [ CPUFloatType{4} ]
```
