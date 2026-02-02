# Sin

Sin

## Usage

``` r
torch_sin(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## sin(input, out=NULL) -\> Tensor

Returns a new tensor with the sine of the elements of `input`.

\$\$ \mbox{out}\_{i} = \sin(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_sin(a)
}
#> torch_tensor
#> -0.9804
#> -0.9425
#>  0.8149
#> -0.7122
#> [ CPUFloatType{4} ]
```
