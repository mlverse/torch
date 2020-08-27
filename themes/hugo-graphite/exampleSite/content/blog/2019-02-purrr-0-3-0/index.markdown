---
title: 'purrr 0.3.0'
author: Lionel Henry, Jenny Bryan, Hadley Wickham, Greg
slug: purrr-0-3-0
date: '2019-02-06'
description: >
  purrr 0.3.0 is now on CRAN.
categories:
  - package
photo:
  url: https://unsplash.com/photos/NodtnCsLdTE
  author: Mikhail Vasilyev
---



We're excited to announce the release of [purrr](https://purrr.tidyverse.org) 0.3.0! purrr enhances R’s functional programming toolkit by providing a complete and consistent set of tools for working with functions and vectors.


```r
# Install the latest version with:
install.packages("purrr")

# Start working with purrr:
library(purrr)
```

In this release, `pluck()` gets a few new variants:

* `pluck<-`, `assign_in()` and `modify_in()` allow deep modifications of nested structures.

* `chuck()` is a stricter variant of `pluck()` that consistently fails instead of returning `NULL` when the structure does not have the expected shape.

A new set of tools helps limiting the rate at which a function is called:

* `slowly()` forces a function to sleep between each invokation.

* `insistently()` automatically invokes a function again on error until it succeeds, and sleeps between invokations.

* The `rate_delay()` and `rate_backoff()` helpers control the invokation rate of `slowly()` and `insistently()`.

The reduce and map functions gain a few improvements:

* `map_if()` accepts an optional function with the `.else` parameter. This function is applied on elements for which the predicate is `FALSE`.

* `map_at()` now accepts `vars()` selections. This lets you use selection helpers like `dplyr::starts_with()` to determine the elements of a list which should be mapped.

* `reduce()` now supports early termination of a computation. Just return a value wrapped in a `done()` to signal to `reduce()` that you're done.

Besides these new functions and tools, purrr 0.3.0 is mostly a polishing release. We have improved the consistency of behaviour:

* `modify()` is now a wrapper around `[[<-` instead of `[<-`. This makes it compatible with a larger variety of S3 vector classes.

* Predicate functions (such that you would pass to `map_if()`) now must return a single `TRUE` or `FALSE`. Missing values and integers are no longer valid predicate outputs.

Finally, we improved the consistency of the interface:

* The direction of iteration/application is now consistently specified with a `.dir` argument.

* Many missing functions were added to fill the gaps: `accumulate2()`, `imodify()`, `map_depth()`, ...

* `partial()` has a much improved and more flexible interface.

Find a detailed account of the changes in the [NEWS](https://github.com/tidyverse/purrr/blob/master/NEWS.md#purrr-030) file.


## New pluck variants

`pluck()` implements a generalised form of `[[` that allow you to index deeply and flexibly into data structures. For instance, `pluck(x, "foo", 2)` is equivalent to `x[["foo"]][[2]]`. You can also supply a default value in case the element does not exist. For instance, `pluck(x, "foo", 2, .default = NA)` is equivalent to `x[["foo"]][[2]]`, returning an `NA` if that element doesn't exist. purrr 0.3.0 introduces variants of `pluck()` to make it easier to work with deep data structures.


### Pluck assignment

This release introduces the new functions `pluck<-`, `assign_in()` and `modify_in()` as assignment variants of `pluck()`. To illustrate deep assignment, let's create a nested data structure:


```r
x <- list(foo = list(1, 2), bar = list(3, 4))
str(x)
#> List of 2
#>  $ foo:List of 2
#>   ..$ : num 1
#>   ..$ : num 2
#>  $ bar:List of 2
#>   ..$ : num 3
#>   ..$ : num 4
```

This sort of repeated structure is the kind of data where `pluck()` shines:


```r
pluck(x, "foo", 2)
#> [1] 2

pluck(x, "bar", 1)
#> [1] 3
```

You can now use the same syntax to modify the data:


```r
pluck(x, "foo", 2) <- 100
str(x)
#> List of 2
#>  $ foo:List of 2
#>   ..$ : num 1
#>   ..$ : num 100
#>  $ bar:List of 2
#>   ..$ : num 3
#>   ..$ : num 4
```

`pluck<-` also has a functional form that does not modify objects in your environment, but instead returns a modified copy:


```r
out <- assign_in(x, list("foo", 2), 2000)

# The object is still the same as before
str(x)
#> List of 2
#>  $ foo:List of 2
#>   ..$ : num 1
#>   ..$ : num 100
#>  $ bar:List of 2
#>   ..$ : num 3
#>   ..$ : num 4

# The modified data is in `out`
str(out)
#> List of 2
#>  $ foo:List of 2
#>   ..$ : num 1
#>   ..$ : num 2000
#>  $ bar:List of 2
#>   ..$ : num 3
#>   ..$ : num 4
```

Finally, `modify_in()` is a variant of `modify()` that only changes the pluck location with the result of applying a function:


```r
out <- modify_in(x, list("foo", 2), as.character)
str(out)
#> List of 2
#>  $ foo:List of 2
#>   ..$ : num 1
#>   ..$ : chr "100"
#>  $ bar:List of 2
#>   ..$ : num 3
#>   ..$ : num 4
```


### Stricter pluck()

Thanks to Daniel Barnett (@daniel-barnett on Github), `pluck()` now has a stricter cousin `chuck()`. Whereas `pluck()` is very permissive regarding non-existing locations and returns `NULL` in these cases, and `[[` inconsistently returns `NULL`, `NA`, or throws an error, `chuck()` fails consistently with informative messages (i.e., it "chucks" an error message):


```r
pluck(list(1), "foo")
#> NULL

chuck(list(1), "foo")
#> Error: Index 1 is attempting to pluck from an unnamed vector using a string name
```


## Rates

Thanks to Richie Cotton (@richierocks) and Ian Lyttle (@ijlyttle), purrr gains a function operator to make a function call itself repeatedly when an error occurs.


```r
counter <- 0

f <- function(...) {
  if (counter < 2) {
    counter <<- counter + 1
    stop("tilt!")
  }
  "result"
}

f()
#> Error in f(): tilt!
```

If the function is wrapped with `insistently()`, it will try a few times before giving up:


```r
# Reset counter
counter <- 0

f2 <- insistently(f)
f2()
#> [1] "result"
```

Another rate limiting function is `slowly()`. While `insistently()` loops by itself, `slowly()` is designed to be used in your own loops, for instance in a map iteration:


```r
f <- function(...) print(Sys.time())

walk(1:3, f)
#> [1] "2019-03-06 12:50:03 PST"
#> [1] "2019-03-06 12:50:03 PST"
#> [1] "2019-03-06 12:50:03 PST"

walk(1:3, slowly(f))
#> [1] "2019-03-06 12:50:03 PST"
#> [1] "2019-03-06 12:50:04 PST"
#> [1] "2019-03-06 12:50:05 PST"
```

`slowly()` uses a constant rate by default while `insistently()` uses a backoff rate. The rate limiting can be configured with optional jitter via `rate_backoff()` and `rate_delay()`, which implement exponential backoff rate and constant rate respectively.




```r
walk(1:3, slowly(f, rate_backoff(2, max_times = Inf)))
#> [1] "2019-03-06 12:50:05 PST"
#> [1] "2019-03-06 12:50:07 PST"
#> [1] "2019-03-06 12:50:10 PST"
```

## Map and reduce improvements

### `map_if()`... or else?

If you like using `map_if()`, perhaps you'll find the new `.else` argument useful. `.else` is a function applied to elements for which the predicate is `FALSE`:


```r
map_if(iris, is.numeric, mean, .else = nlevels)
#> $Sepal.Length
#> [1] 5.843333
#> 
#> $Sepal.Width
#> [1] 3.057333
#> 
#> $Petal.Length
#> [1] 3.758
#> 
#> $Petal.Width
#> [1] 1.199333
#> 
#> $Species
#> [1] 3
```


### New `map_at()` features

Colin Fay (@ColinFay) has added support for tidyselect expressions to `map_at()` and other `_at` mappers. This brings the interface of these functions closer to scoped functions from the dplyr package, such as `dplyr::mutate_at()`. Note that `vars()` is currently not reexported from purrr, so you need to use `dplyr::vars()` or `ggplot2::vars()` for the time being.


```r
suppressMessages(library("dplyr"))

x <- list(
  foo = 1:5,
  bar = 6:10,
  baz = 11:15
)

map_at(x, vars(starts_with("b")), mean)
#> $foo
#> [1] 1 2 3 4 5
#> 
#> $bar
#> [1] 8
#> 
#> $baz
#> [1] 13
```

`map_at()` now also supports negative selections:


```r
map_at(x, -2, `*`, 1000)
#> $foo
#> [1] 1000 2000 3000 4000 5000
#> 
#> $bar
#> [1]  6  7  8  9 10
#> 
#> $baz
#> [1] 11000 12000 13000 14000 15000
```


### Early termination of reduction

`reduce()` is an operation that combines the elements of a vector into a single value by calling a binary function repeatedly with the result so far and the next input of a vector. `reduce()` and its variant `accumulate()` now support early termination of the reduction. To halt the computation, just return the last value wrapped in a `done()` box:


```r
# This computes the total sum of the input vector
reduce(1:100, ~ .x + .y)
#> [1] 5050

# This stops as soon as the sum is greater than 50
reduce(1:100, ~ if (.x > 50) done(.x) else .x + .y)
#> [1] 55
```

This feature takes inspiration from the [Clojure](https://clojuredocs.org/clojure.core/reduced) language.


## Consistency

In this polishing release, a lot of effort went towards consistency of behaviour and of the interface.


### Behaviour

#### Better support for S3 vectors

We are working hard on improving support for S3 vectors in the tidyverse. As of this release, `modify()` is now a wrapper around `[[<-` instead of `[<-`. This should make it directly compatible with a larger set of vector classes. Thanks to the work of Mikko Marttila (@mikmart), `pmap()` and `pwalk()` also do a better job of preserving S3 classes. Finally, `pluck()` now properly calls the `[[` methods of S3 objects.

In the next version of purrr, we plan to use the in-development vctrs package to provide more principled and predictable vector operations. This should help us preserve the class and properties of S3 vectors like factors, dates, or your custom classes.


#### Stricter predicate checking

purrr now checks the results of your predicate functions, which must now consistently return `TRUE` or `FALSE`. We no longer offer support for `NA` or for boolish numeric values (R normally interprets 0 as `FALSE` and all other values as `TRUE`). The purpose of this change is to detect errors earlier with a more relevant error message.


```r
keep(c(1, NA, 3), ~ . %% 2 == 0)
#> Error: Predicate functions must return a single `TRUE` or `FALSE`, not a missing value
```


### Interface

#### Direction of application

The direction of application is now specified the same way across purrr functions. `reduce()`, `compose()` and `detect()` now have a `.dir` parameter that can take the value `"forward"` or `"backward"`. This terminology should be less ambiguous than "left" and "right":


```r
reduce(1:4, `-`, .dir = "backward")

compose(foo, bar, .dir = "forward")

detect(1:5, ~ . %% 2 == 0, .dir = "backward")
```

Note that the backward version of `reduce()` (called right-reduce in the literature) applies the reduced function in a slightly different way than `reduce_right()`. The new algorithm is more consistent with how this operation is usually defined in other languages.

Following the introduction of the `.dir` parameters, the `_right` variants such as `reduce_right()` have been soft-deprecated, as well as the `.right` parameter of `detect()` and `detect_index()`.


#### partial()

`partial()` has been rewritten to be a simple wrapper around `call_modify()` and `eval_tidy()` from the rlang package. Consequently, the `.env`, `.lazy` and `.first` arguments are soft-deprecated and replaced by a flexible syntax.

To control the timing of evaluation, unquote the partialised arguments that should be evaluated only once when the function is created. The non-unquoted arguments are evaluated at each invokation of the function:


```r
my_list <- partial(list, lazy = rnorm(3), eager = !!rnorm(3))

my_list()
#> $lazy
#> [1]  0.2945451  0.3897943 -1.2080762
#> 
#> $eager
#> [1] -0.1842525 -1.3713305 -0.5991677

my_list()
#> $lazy
#> [1] -0.3636760 -1.6266727 -0.2564784
#> 
#> $eager
#> [1] -0.1842525 -1.3713305 -0.5991677
```

You can also control the position of the future arguments by passing an empty `... = ` parameter. This syntax is powered by `rlang::call_modify()` and allows you to add or move dots in a quoted function call. In the case of `partial()`, the dots represent the future arguments. We use this syntax in the following snippet to position the future arguments right between two partialised arguments:


```r
my_list <- partial(list, 1, ... = , 2)

my_list()
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] 2

my_list("foo")
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] "foo"
#> 
#> [[3]]
#> [1] 2
```


#### `exec()` replaces `invoke()`

We are retiring `invoke()` and `invoke_map()` in favour of `exec()`. Retirement means that we'll keep these functions indefinitely in the package, but we won't add features or recommend using them.

We are now favouring `exec()`, which uses the tidy dots syntax for passing lists of arguments:


```r
# Before:
invoke(mean, list(na.rm = TRUE), x = 1:10)
#> [1] 5.5

# After
exec(mean, 1:10, !!!list(na.rm = TRUE))
#> [1] 5.5
```


#### Filling the missing parts

* purrr 0.3.0 introduces `accumulate2()`, `modify2()` and `imodify()` variants.

* By popular request, `at_depth()` is back as `map_depth()`. Unlike `modify_depth()` which preserves the class structure of the input tree, this variant only returns trees made of lists of lists (up to the given depth), coercing vectors if needed.


## Thanks!

Thanks to all the contributors for this release!

  [&#xFF20;ArtemSokolov](https://github.com/ArtemSokolov), [&#xFF20;batpigandme](https://github.com/batpigandme), [&#xFF20;bbrewington](https://github.com/bbrewington), [&#xFF20;billdenney](https://github.com/billdenney), [&#xFF20;cderv](https://github.com/cderv), [&#xFF20;cfhammill](https://github.com/cfhammill), [&#xFF20;ColinFay](https://github.com/ColinFay), [&#xFF20;dan-reznik](https://github.com/dan-reznik), [&#xFF20;daniel-barnett](https://github.com/daniel-barnett), [&#xFF20;danilinares](https://github.com/danilinares), [&#xFF20;drtjc](https://github.com/drtjc), [&#xFF20;egnha](https://github.com/egnha), [&#xFF20;Eluvias](https://github.com/Eluvias), [&#xFF20;flying-sheep](https://github.com/flying-sheep), [&#xFF20;gergness](https://github.com/gergness), [&#xFF20;gvwilson](https://github.com/gvwilson), [&#xFF20;hadley](https://github.com/hadley), [&#xFF20;hammer](https://github.com/hammer), [&#xFF20;ijlyttle](https://github.com/ijlyttle), [&#xFF20;ilarischeinin](https://github.com/ilarischeinin), [&#xFF20;IndrajeetPatil](https://github.com/IndrajeetPatil), [&#xFF20;ISPritchin](https://github.com/ISPritchin), [&#xFF20;jameslairdsmith](https://github.com/jameslairdsmith), [&#xFF20;jameslamb](https://github.com/jameslamb), [&#xFF20;jrnold](https://github.com/jrnold), [&#xFF20;kcf-jackson](https://github.com/kcf-jackson), [&#xFF20;leungi](https://github.com/leungi), [&#xFF20;lionel-](https://github.com/lionel-), [&#xFF20;loladze](https://github.com/loladze), [&#xFF20;maxheld83](https://github.com/maxheld83), [&#xFF20;mikmart](https://github.com/mikmart), [&#xFF20;MilesMcBain](https://github.com/MilesMcBain), [&#xFF20;moodymudskipper](https://github.com/moodymudskipper), [&#xFF20;mrstlee](https://github.com/mrstlee), [&#xFF20;namelessjon](https://github.com/namelessjon), [&#xFF20;r-cheologist](https://github.com/r-cheologist), [&#xFF20;randomgambit](https://github.com/randomgambit), [&#xFF20;rmflight](https://github.com/rmflight), [&#xFF20;roumail](https://github.com/roumail), [&#xFF20;Ryo-N7](https://github.com/Ryo-N7), [&#xFF20;serina-robinson](https://github.com/serina-robinson), [&#xFF20;skaltman](https://github.com/skaltman), [&#xFF20;suraggupta](https://github.com/suraggupta), [&#xFF20;thays42](https://github.com/thays42), [&#xFF20;tyluRp](https://github.com/tyluRp), [&#xFF20;tzakharko](https://github.com/tzakharko), [&#xFF20;VincentGuyader](https://github.com/VincentGuyader), [&#xFF20;wlandau](https://github.com/wlandau), [&#xFF20;wmayner](https://github.com/wmayner), [&#xFF20;yanxianl](https://github.com/yanxianl), [&#xFF20;yutannihilation](https://github.com/yutannihilation), and [&#xFF20;yysh12](https://github.com/yysh12)
