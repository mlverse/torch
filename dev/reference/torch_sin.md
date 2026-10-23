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
#> -0.2822
#>  0.8484
#>  0.3922
#>  0.9887
#> [ CPUFloatType{4} ]
```
