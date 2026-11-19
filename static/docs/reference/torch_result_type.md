# Result_type

Result_type

## Usage

``` r
torch_result_type(tensor1, tensor2)
```

## Arguments

- tensor1:

  (Tensor or Number) an input tensor or number

- tensor2:

  (Tensor or Number) an input tensor or number

## result_type(tensor1, tensor2) -\> dtype

Returns the `torch_dtype` that would result from performing an
arithmetic operation on the provided input tensors. See type promotion
documentation for more information on the type promotion logic.

## Examples

``` r
if (torch_is_installed()) {

torch_result_type(tensor1 = torch_tensor(c(1, 2), dtype=torch_int()), tensor2 = 1)
}
#> torch_Float
```
