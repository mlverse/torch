# Abstract Base Class for LibTorch Optimizers

Abstract base class for wrapping LibTorch C++ optimizers.

## Usage

``` r
optimizer_ignite(
  name = NULL,
  ...,
  private = NULL,
  active = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- name:

  (optional) name of the optimizer

- ...:

  Pass any number of fields or methods. You should at least define the
  `initialize` and `step` methods. See the examples section.

- private:

  (optional) a list of private methods for the optimizer.

- active:

  (optional) a list of active methods for the optimizer.

- parent_env:

  used to capture the right environment to define the class. The default
  is fine for most situations.
