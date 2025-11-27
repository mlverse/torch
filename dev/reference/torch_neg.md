# Neg

Neg

## Usage

``` r
torch_neg(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## neg(input, out=NULL) -\> Tensor

Returns a new tensor with the negative of the elements of `input`.

\$\$ \mbox{out} = -1 \times \mbox{input} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(5))
a
torch_neg(a)
}
#> torch_tensor
#> -0.9658
#> -0.9673
#> -0.9069
#>  0.4794
#> -0.2410
#> [ CPUFloatType{5} ]
```
