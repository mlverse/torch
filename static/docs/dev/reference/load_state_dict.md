# Load a state dict file

This function should only be used to load models saved in python. For it
to work correctly you need to use `torch.save` with the flag:
`_use_new_zipfile_serialization=True` and also remove all `nn.Parameter`
classes from the tensors in the dict.

## Usage

``` r
load_state_dict(path, ..., legacy_stream = FALSE)
```

## Arguments

- path:

  to the state dict file

- ...:

  additional arguments that are currently not used.

- legacy_stream:

  if `TRUE` then the state dict is loaded using a a legacy way of
  handling streams.

## Value

a named list of tensors.

## Details

The above might change with development of
[this](https://github.com/pytorch/pytorch/issues/37213) in pytorch's C++
api.
