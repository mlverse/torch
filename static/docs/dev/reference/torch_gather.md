# Gather

Gather

## Usage

``` r
torch_gather(self, dim, index, sparse_grad = FALSE)
```

## Arguments

- self:

  (Tensor) the source tensor

- dim:

  (int) the axis along which to index

- index:

  (LongTensor) the indices of elements to gather

- sparse_grad:

  (bool,optional) If `TRUE`, gradient w.r.t. `input` will be a sparse
  tensor.

## gather(input, dim, index, sparse_grad=FALSE) -\> Tensor

Gathers values along an axis specified by `dim`.

For a 3-D tensor the output is specified by::

    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

If `input` is an n-dimensional tensor with size \\(x_0, x_1...,
x\_{i-1}, x_i, x\_{i+1}, ..., x\_{n-1})\\ and `dim = i`, then `index`
must be an \\n\\-dimensional tensor with size \\(x_0, x_1, ...,
x\_{i-1}, y, x\_{i+1}, ..., x\_{n-1})\\ where \\y \geq 1\\ and `out`
will have the same size as `index`.

## Examples

``` r
if (torch_is_installed()) {

t = torch_tensor(matrix(c(1,2,3,4), ncol = 2, byrow = TRUE))
torch_gather(t, 2, torch_tensor(matrix(c(1,1,2,1), ncol = 2, byrow=TRUE), dtype = torch_int64()))
}
#> torch_tensor
#>  1  1
#>  4  3
#> [ CPUFloatType{2,2} ]
```
