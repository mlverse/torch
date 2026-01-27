# Poisson

Poisson

## Usage

``` r
torch_poisson(self, generator = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor containing the rates of the Poisson
  distribution

- generator:

  (`torch.Generator`, optional) a pseudorandom number generator for
  sampling

## poisson(input \*, generator=NULL) -\> Tensor

Returns a tensor of the same size as `input` with each element sampled
from a Poisson distribution with rate parameter given by the
corresponding element in `input` i.e.,

\$\$ \mbox{out}\_i \sim \mbox{Poisson}(\mbox{input}\_i) \$\$

## Examples

``` r
if (torch_is_installed()) {

rates = torch_rand(c(4, 4)) * 5  # rate parameter between 0 and 5
torch_poisson(rates)
}
#> torch_tensor
#>  3  3  0  4
#>  0  4  0  0
#>  1  5  1  4
#>  3  2  1  7
#> [ CPUFloatType{4,4} ]
```
