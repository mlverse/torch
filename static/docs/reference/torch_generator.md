# Create a Generator object

A `torch_generator` is an object which manages the state of the
algorithm that produces pseudo random numbers. Used as a keyword
argument in many In-place random sampling functions.

## Usage

``` r
torch_generator()
```

## Examples

``` r
if (torch_is_installed()) {

# Via string
generator <- torch_generator()
generator$current_seed()
generator$set_current_seed(1234567L)
generator$current_seed()

}
#> integer64
#> [1] 1234567
```
