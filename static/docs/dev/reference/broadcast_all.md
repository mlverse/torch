# Given a list of values (possibly containing numbers), returns a list where each value is broadcasted based on the following rules:

Raises value_error: if any of the values is not a `numeric` instance, a
`torch.*Tensor` instance, or an instance implementing **torch_function**
TODO: add has_torch_function((v,)) See:
https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py

## Usage

``` r
broadcast_all(values)
```

## Arguments

- values:

  List of:

  - `torch.*Tensor` instances are broadcasted as per
    `_broadcasting-semantics`.

  - `numeric` instances (scalars) are upcast to tensors having the same
    size and type as the first tensor passed to `values`. If all the
    values are scalars, then they are upcasted to scalar Tensors. values
    (list of `numeric`, `torch.*Tensor` or objects implementing
    **torch_function**)
