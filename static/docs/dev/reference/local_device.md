# Device contexts

Device contexts

## Usage

``` r
local_device(device, ..., .env = parent.frame())

with_device(code, ..., device)
```

## Arguments

- device:

  A torch device to be used by default when creating new tensors.

- ...:

  currently unused.

- .env:

  The environment to use for scoping.

- code:

  The code to be evaluated in the modified environment.

## Functions

- `with_device()`: Modifies the default device for the selected context.
