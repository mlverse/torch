# Sub

Sub

## Usage

``` r
torch_sub(self, other, alpha = 1L)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor or Scalar) the tensor or scalar to subtract from `input`

- alpha:

  the scalar multiplier for other

## sub(input, other, \*, alpha=1, out=None) -\> Tensor

Subtracts `other`, scaled by `alpha`, from `input`.

\$\$ \mbox{{out}}\_i = \mbox{{input}}\_i - \mbox{{alpha}} \times
\mbox{{other}}\_i \$\$

Supports broadcasting to a common shape , type promotion , and integer,
float, and complex inputs.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2))
b <- torch_tensor(c(0, 1))
torch_sub(a, b, alpha=2)
}
#> torch_tensor
#>  1
#>  0
#> [ CPUFloatType{2} ]
```
