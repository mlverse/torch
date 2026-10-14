# Exp

Exp

## Usage

``` r
torch_exp(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## exp(input, out=NULL) -\> Tensor

Returns a new tensor with the exponential of the elements of the input
tensor `input`.

\$\$ y\_{i} = e^{x\_{i}} \$\$

## Examples

``` r
if (torch_is_installed()) {

torch_exp(torch_tensor(c(0, log(2))))
}
#> torch_tensor
#>  1
#>  2
#> [ CPUFloatType{2} ]
```
