# Adds the 'jit_tuple' class to the input

Allows specifying that an output or input must be considered a jit tuple
and instead of a list or dictionary when tracing.

## Usage

``` r
jit_tuple(x)
```

## Arguments

- x:

  the list object that will be converted to a tuple.
