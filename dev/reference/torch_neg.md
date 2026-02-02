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
#> -0.6010
#> -0.3329
#> -0.0180
#> -0.9933
#>  1.1041
#> [ CPUFloatType{5} ]
```
