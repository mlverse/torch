# Creates an iterable dataset

Creates an iterable dataset

## Usage

``` r
iterable_dataset(
  name,
  inherit = IterableDataset,
  ...,
  private = NULL,
  active = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- name:

  a name for the dataset. It it's also used as the class for it.

- inherit:

  you can optionally inherit from a dataset when creating a new dataset.

- ...:

  public methods for the dataset class

- private:

  passed to
  [`R6::R6Class()`](https://r6.r-lib.org/reference/R6Class.html).

- active:

  passed to
  [`R6::R6Class()`](https://r6.r-lib.org/reference/R6Class.html).

- parent_env:

  An environment to use as the parent of newly-created objects.

## Examples

``` r
if (torch_is_installed()) {
ids <- iterable_dataset(
  name = "hello",
  initialize = function(n = 5) {
    self$n <- n
    self$i <- 0
  },
  .iter = function() {
    i <- 0
    function() {
      i <<- i + 1
      if (i > self$n) {
        coro::exhausted()
      } else {
        i
      }
    }
  }
)
coro::collect(ids()$.iter())
}
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] 2
#> 
#> [[3]]
#> [1] 3
#> 
#> [[4]]
#> [1] 4
#> 
#> [[5]]
#> [1] 5
#> 
```
