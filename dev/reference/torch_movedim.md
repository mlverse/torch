# Movedim

Movedim

## Usage

``` r
torch_movedim(self, source, destination)
```

## Arguments

- self:

  (Tensor) the input tensor.

- source:

  (int or tuple of ints) Original positions of the dims to move. These
  must be unique.

- destination:

  (int or tuple of ints) Destination positions for each of the original
  dims. These must also be unique.

## movedim(input, source, destination) -\> Tensor

Moves the dimension(s) of `input` at the position(s) in `source` to the
position(s) in `destination`.

Other dimensions of `input` that are not explicitly moved remain in
their original order and appear at the positions not specified in
`destination`.

## Examples

``` r
if (torch_is_installed()) {

t <- torch_randn(c(3,2,1))
t
torch_movedim(t, 2, 1)$shape
torch_movedim(t, 2, 1)
torch_movedim(t, c(2, 3), c(1, 2))$shape
torch_movedim(t, c(2, 3), c(1, 2))
}
#> torch_tensor
#> (1,.,.) = 
#>  -0.7080  0.4082  0.4454
#> 
#> (2,.,.) = 
#>  -0.8413  0.4686 -0.7510
#> [ CPUFloatType{2,1,3} ]
```
