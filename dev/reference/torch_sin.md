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
#> -0.2617
#> -0.7686
#> -0.3192
#> -0.3385
#> [ CPUFloatType{4} ]
```
