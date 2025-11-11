# I0

I0

## Usage

``` r
torch_i0(self)
```

## Arguments

- self:

  (Tensor) the input tensor

## i0(input, \*, out=None) -\> Tensor

Computes the zeroth order modified Bessel function of the first kind for
each element of `input`.

\$\$ \mbox{out}\_{i} = I_0(\mbox{input}\_{i}) = \sum\_{k=0}^{\infty}
\frac{(\mbox{input}\_{i}^2/4)^k}{(k!)^2} \$\$

## Examples

``` r
if (torch_is_installed()) {

torch_i0(torch_arange(start = 0, end = 5, dtype=torch_float32()))
}
#> torch_tensor
#>   1.0000
#>   1.2661
#>   2.2796
#>   4.8808
#>  11.3019
#>  27.2399
#> [ CPUFloatType{6} ]
```
