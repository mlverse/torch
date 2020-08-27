---
title: slider 0.1.0
author: Davis Vaughan
date: '2020-02-10'
slug: slider-0-1-0
description: 
  slider 0.1.0 is now available on CRAN. It provides a family of general purpose
  sliding window functions.
categories:
  - package
tags:
  - slider
photo: 
  url: https://unsplash.com/photos/6RLa-tC7y0I
  author: Arie Wubben
---



I'm thrilled to announce that the first version of [slider](https://davisvaughan.github.io/slider/) is now available on CRAN!

slider provides a family of general purpose sliding window functions, which can be used to compute moving averages, cumulatives sums, rolling regressions, and any other sliding operation.

This package is a combination of ideas from a variety of sources, including: 

- [purrr](https://purrr.tidyverse.org/) for the overall package API

- [SQL's window functions](https://www.postgresql.org/docs/9.1/sql-expressions.html#SYNTAX-WINDOW-FUNCTIONS) for the argument API

- [Earo Wang's `tsibble::slide()`](https://github.com/tidyverts/tsibble) for the function names

- [data.table's non-equi joins](https://rdatatable.gitlab.io/data.table/) for inspiration on how `slide_index()` should work

Install slider with:


```r
install.packages("slider")
```

This blog post summarizes the three key functions in slider: `slide()`, `slide_index()`, and `slide_period()`.


```r
library(slider)
library(tibble)
library(purrr)
library(lubridate, warn.conflicts = FALSE)
library(dplyr, warn.conflicts = FALSE)
```


## Sliding

`purrr::map()` allows you to iterate over a vector one element at a time and apply a function to each element. `slide()` takes that concept and generalizes it so that you can iterate over _sliding windows_ of a vector, applying any function that you want to each window. To start exploring this idea, note that the defaults of `slide()` are essentially identical to `map()`.


```r
# A vector of sales data for our business
sales_vec <- c(2, 4, 3, 5)

slide(sales_vec, ~.x)
#> [[1]]
#> [1] 2
#> 
#> [[2]]
#> [1] 4
#> 
#> [[3]]
#> [1] 3
#> 
#> [[4]]
#> [1] 5
```

Things get more interesting when you consider the additional arguments to `slide()`:

- `.before`: How many elements _before_ the current one should be included in the window?

- `.after`: How many elements _after_ the current one should be included in the window?

- `.complete`: Should `.f` only be evaluated when there is enough data to make a complete window?

- `.step`: The number of elements to shift forward between calls to `.f`.

By setting `.before = 1` we can construct moving windows along `.x`, adding the current element and the one before it into the window.


```r
slide(sales_vec, ~.x, .before = 1)
#> [[1]]
#> [1] 2
#> 
#> [[2]]
#> [1] 2 4
#> 
#> [[3]]
#> [1] 4 3
#> 
#> [[4]]
#> [1] 3 5
```

Notice how in the first result our slice just has one element. This is an _incomplete_ window because it isn't possible to look one element before the first element. By default, `slide()` computes `.f` on incomplete windows, but you can force it to only be computed on complete windows with `.complete`.


```r
slide(sales_vec, ~.x, .before = 1, .complete = TRUE)
#> [[1]]
#> NULL
#> 
#> [[2]]
#> [1] 2 4
#> 
#> [[3]]
#> [1] 4 3
#> 
#> [[4]]
#> [1] 3 5
```

The API of slider is intentionally very similar to purrr. `slide()` always returns a list (like `map()`), and the size of the result is always the same size as the input. As you might have guessed, there are also suffixed versions available to return more specific output. Say we want to compute a 2-value moving average from our sales. We might use:


```r
slide_dbl(sales_vec, mean, .before = 1)
#> [1] 2.0 3.0 3.5 4.0
```

There is also a new suffix, `*_vec()`, which attempts to automatically simplify the results using the type rules provided by [vctrs](https://vctrs.r-lib.org/).


```r
slide_vec(sales_vec, mean, .before = 1)
#> [1] 2.0 3.0 3.5 4.0

slide_vec(sales_vec, ~mean(.x) > 3, .before = 1)
#> [1] FALSE FALSE  TRUE  TRUE
```

Lastly, the one big difference between how `slide()` and `map()` iterate over vectors is how they treat data frames. To `map()`, a data frame is a vector of columns, to `slide()` it is a [vector of rows](https://blog.davisvaughan.com/2019/10/16/data-frames-as-vectors-of-rows/). In a way, this makes `slide()` a generic row-wise iterator over data frames.



```r
index_vec <- as.Date("2019-08-29") + c(0, 1, 5, 6)
wday_vec <- as.character(wday(index_vec, label = TRUE))

company <- tibble(
  sales = sales_vec,
  index = index_vec,
  wday = wday_vec
)

company
#> # A tibble: 4 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     2 2019-08-29 Thu  
#> 2     4 2019-08-30 Fri  
#> 3     3 2019-09-03 Tue  
#> 4     5 2019-09-04 Wed
```

Over columns:


```r
map(company, ~.x)
#> $sales
#> [1] 2 4 3 5
#> 
#> $index
#> [1] "2019-08-29" "2019-08-30" "2019-09-03" "2019-09-04"
#> 
#> $wday
#> [1] "Thu" "Fri" "Tue" "Wed"
```

Over rows:


```r
slide(company, ~.x)
#> [[1]]
#> # A tibble: 1 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     2 2019-08-29 Thu  
#> 
#> [[2]]
#> # A tibble: 1 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     4 2019-08-30 Fri  
#> 
#> [[3]]
#> # A tibble: 1 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     3 2019-09-03 Tue  
#> 
#> [[4]]
#> # A tibble: 1 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     5 2019-09-04 Wed
```

You can also still use the additional arguments to construct sliding windows along your data frame.


```r
slide(company, ~.x, .before = 2)
#> [[1]]
#> # A tibble: 1 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     2 2019-08-29 Thu  
#> 
#> [[2]]
#> # A tibble: 2 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     2 2019-08-29 Thu  
#> 2     4 2019-08-30 Fri  
#> 
#> [[3]]
#> # A tibble: 3 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     2 2019-08-29 Thu  
#> 2     4 2019-08-30 Fri  
#> 3     3 2019-09-03 Tue  
#> 
#> [[4]]
#> # A tibble: 3 x 3
#>   sales index      wday 
#>   <dbl> <date>     <chr>
#> 1     4 2019-08-30 Fri  
#> 2     3 2019-09-03 Tue  
#> 3     5 2019-09-04 Wed
```

## Index sliding

Throughout R's history, a few of the features of `slide()` have been available in other packages. You could accomplish some of the same things with `zoo::rollapply()`, `tsibble::slide()`, and even with my original attempt at this, `tibbletime::rollify()`.

But none of these methods ever solved a problem that is incredibly common in business-oriented data analysis. What happens when you have a date index attached to when the sales happened, and you want to compute a moving average over _two days_? `slide()` doesn't solve this problem either, because you might have date gaps in your vector that it doesn't know about. To demonstrate this, let's assume that you are interested in computing those two day windows. To visualize them, we'll print out the week day that would be associated with these naive windows if we used `slide()`.


```r
wday_vec
#> [1] "Thu" "Fri" "Tue" "Wed"

slide(wday_vec, ~.x, .before = 1)
#> [[1]]
#> [1] "Thu"
#> 
#> [[2]]
#> [1] "Thu" "Fri"
#> 
#> [[3]]
#> [1] "Fri" "Tue"
#> 
#> [[4]]
#> [1] "Tue" "Wed"
```

Notice the third window! We started on Tuesday, which means the window should only include `[Mon, Tue]`, but Friday is also included here. This happens because `slide()` looks back a number of _values_, and knows nothing about how to compute this `[Mon, Tue]` _range_ to slide between. This differentiation between values and ranges comes from SQL, and is further explained in a bit more detail by [Vertica's window function documentation](https://www.vertica.com/docs/9.2.x/HTML/Content/Authoring/SQLReferenceManual/Functions/Analytic/window_frame_clause.htm?origin_team=T02V9CHFH#ROWSversusRANGE).

To solve this specific problem of sliding with respect to an index, we'll need a new function, `slide_index()`. It's similar to `slide()`, and has all of the same suffixed versions, but allows you to pass in a secondary index to slide relative to. Ranges to slide between are computed as `.i - .before` and `.i + .after`, meaning that you can use more interesting objects for `.before`, like `lubridate::days()`! It just has to implement `-` and `+` methods when combined with your index.


```r
wday_vec
#> [1] "Thu" "Fri" "Tue" "Wed"
index_vec
#> [1] "2019-08-29" "2019-08-30" "2019-09-03" "2019-09-04"

slide_index(.x = wday_vec, .i = index_vec, ~.x, .before = days(1))
#> [[1]]
#> [1] "Thu"
#> 
#> [[2]]
#> [1] "Thu" "Fri"
#> 
#> [[3]]
#> [1] "Tue"
#> 
#> [[4]]
#> [1] "Tue" "Wed"
```

This correctly buckets Tuesday in its own group, since there is no data point for the Monday before it. We can compare the difference between a two-value vs a two-day moving average like so:


```r
company %>%
  mutate(
    two_value = slide_dbl(sales, mean, .before = 1),
    two_day = slide_index_dbl(sales, index, mean, .before = days(1)),
  )
#> # A tibble: 4 x 5
#>   sales index      wday  two_value two_day
#>   <dbl> <date>     <chr>     <dbl>   <dbl>
#> 1     2 2019-08-29 Thu         2         2
#> 2     4 2019-08-30 Fri         3         3
#> 3     3 2019-09-03 Tue         3.5       3
#> 4     5 2019-09-04 Wed         4         4
```

## Period sliding

While `slide()` and `slide_index()` are great because they are size-stable, sometimes you'll want to take data that has a daily index, break it into monthly chunks, and return results at the monthly level. This implies returning an output that has a different size from your input. To power these ideas, you can use `slide_period()`, which takes an index and a period to chunk by, and then iterates over `.x` relative to those period chunks.

Say we want to take `big_company` below, break it into monthly chunks, and compute and return just the monthly totals.


```r
big_index_vec <- c(
  as.Date("2019-08-30") + 0:4,
  as.Date("2019-11-30") + 0:4
)

big_sales_vec <- c(2, 4, 6, 2, 8, 10, 9, 3, 5, 2)

big_company <- tibble(
  sales = big_sales_vec,
  index = big_index_vec
)

big_company
#> # A tibble: 10 x 2
#>    sales index     
#>    <dbl> <date>    
#>  1     2 2019-08-30
#>  2     4 2019-08-31
#>  3     6 2019-09-01
#>  4     2 2019-09-02
#>  5     8 2019-09-03
#>  6    10 2019-11-30
#>  7     9 2019-12-01
#>  8     3 2019-12-02
#>  9     5 2019-12-03
#> 10     2 2019-12-04
```

`slide_period()` allows you to iterate over your data frame in these monthly chunks, applying whatever function you want to each one.


```r
slide_period(big_company, big_company$index, "month", ~.x)
#> [[1]]
#> # A tibble: 2 x 2
#>   sales index     
#>   <dbl> <date>    
#> 1     2 2019-08-30
#> 2     4 2019-08-31
#> 
#> [[2]]
#> # A tibble: 3 x 2
#>   sales index     
#>   <dbl> <date>    
#> 1     6 2019-09-01
#> 2     2 2019-09-02
#> 3     8 2019-09-03
#> 
#> [[3]]
#> # A tibble: 1 x 2
#>   sales index     
#>   <dbl> <date>    
#> 1    10 2019-11-30
#> 
#> [[4]]
#> # A tibble: 4 x 2
#>   sales index     
#>   <dbl> <date>    
#> 1     9 2019-12-01
#> 2     3 2019-12-02
#> 3     5 2019-12-03
#> 4     2 2019-12-04
```

I find it easiest to wrap up what you want to do into a function, and then apply that to each slice.


```r
monthly_summary <- function(data) {
  summarise(data, start = min(index), end = max(index), total_sales = sum(sales))
}

slide_period_dfr(
  big_company,
  big_company$index,
  "month",
  monthly_summary
)
#> # A tibble: 4 x 3
#>   start      end        total_sales
#>   <date>     <date>           <dbl>
#> 1 2019-08-30 2019-08-31           6
#> 2 2019-09-01 2019-09-03          16
#> 3 2019-11-30 2019-11-30          10
#> 4 2019-12-01 2019-12-04          19
```

Now, you might recognize that you can do this with dplyr:


```r
big_company %>%
  mutate(monthly = floor_date(index, "month")) %>%
  group_by(monthly) %>%
  summarise(sales = sum(sales))
#> # A tibble: 4 x 2
#>   monthly    sales
#>   <date>     <dbl>
#> 1 2019-08-01     6
#> 2 2019-09-01    16
#> 3 2019-11-01    10
#> 4 2019-12-01    19
```

But what you can't easily do is slide over multiple monthly chunks at once, effectively creating a rolling monthly total, from daily data. With `slide_period()`, `.before` works at the period level, so you get to control how many monthly periods are included in your sliding window. Notice how the start dates below go back into the previous month (but only if there was data for it).


```r
slide_period_dfr(
  big_company,
  big_company$index,
  "month",
  monthly_summary,
  .before = 1
)
#> # A tibble: 4 x 3
#>   start      end        total_sales
#>   <date>     <date>           <dbl>
#> 1 2019-08-30 2019-08-31           6
#> 2 2019-08-30 2019-09-03          22
#> 3 2019-11-30 2019-11-30          10
#> 4 2019-11-30 2019-12-04          29
```

## Acknowledgements

A big thanks to some of the early adopters of slider!

[&#x0040;AlanFeder](https://github.com/AlanFeder), [&#x0040;AlunHewinson](https://github.com/AlunHewinson), [&#x0040;echasnovski](https://github.com/echasnovski), [&#x0040;mik3y64](https://github.com/mik3y64), [&#x0040;perlatex](https://github.com/perlatex), and [&#x0040;RobertMyles](https://github.com/RobertMyles).
