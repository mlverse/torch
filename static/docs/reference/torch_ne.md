# Ne

Ne

## Usage

``` r
torch_ne(self, other)
```

## Arguments

- self:

  (Tensor) the tensor to compare

- other:

  (Tensor or float) the tensor or value to compare

## ne(input, other, out=NULL) -\> Tensor

Computes \\input \neq other\\ element-wise.

The second argument can be a number or a tensor whose shape is
broadcastable with the first argument.

## Examples

``` r
if (torch_is_installed()) {

torch_ne(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
         torch_tensor(matrix(rep(c(1,4), each = 2), ncol = 2, byrow=TRUE)))
}
#> torch_tensor
#>  0  1
#>  1  0
#> [ CPUBoolType{2,2} ]
```
