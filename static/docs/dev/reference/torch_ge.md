# Ge

Ge

## Usage

``` r
torch_ge(self, other)
```

## Arguments

- self:

  (Tensor) the tensor to compare

- other:

  (Tensor or float) the tensor or value to compare

## ge(input, other, out=NULL) -\> Tensor

Computes \\\mbox{input} \geq \mbox{other}\\ element-wise.

The second argument can be a number or a tensor whose shape is
broadcastable with the first argument.

## Examples

``` r
if (torch_is_installed()) {

torch_ge(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
         torch_tensor(matrix(c(1,1,4,4), ncol = 2, byrow=TRUE)))
}
#> torch_tensor
#>  1  1
#>  0  1
#> [ CPUBoolType{2,2} ]
```
