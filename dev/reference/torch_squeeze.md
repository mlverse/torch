# Squeeze

Squeeze

## Usage

``` r
torch_squeeze(self, dim)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int, optional) if given, the input will be squeezed only in this
  dimension

## Note

The returned tensor shares the storage with the input tensor, so
changing the contents of one will change the contents of the other.

## squeeze(input, dim=NULL, out=NULL) -\> Tensor

Returns a tensor with all the dimensions of `input` of size `1` removed.

For example, if `input` is of shape: \\(A \times 1 \times B \times C
\times 1 \times D)\\ then the `out` tensor will be of shape: \\(A \times
B \times C \times D)\\.

When `dim` is given, a squeeze operation is done only in the given
dimension. If `input` is of shape: \\(A \times 1 \times B)\\,
`squeeze(input, 0)` leaves the tensor unchanged, but `squeeze(input, 1)`
will squeeze the tensor to the shape \\(A \times B)\\.

## Examples

``` r
if (torch_is_installed()) {

x = torch_zeros(c(2, 1, 2, 1, 2))
x
y = torch_squeeze(x)
y
y = torch_squeeze(x, 1)
y
y = torch_squeeze(x, 2)
y
}
#> torch_tensor
#> (1,1,.,.) = 
#>   0  0
#> 
#> (2,1,.,.) = 
#>   0  0
#> 
#> (1,2,.,.) = 
#>   0  0
#> 
#> (2,2,.,.) = 
#>   0  0
#> [ CPUFloatType{2,2,1,2} ]
```
