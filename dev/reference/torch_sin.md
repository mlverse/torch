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
#>  0.9856
#> -0.2223
#>  0.9384
#>  0.9947
#> [ CPUFloatType{4} ]
```
