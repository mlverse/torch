---
title: stringr 1.4.0
author: [mine, mara Averick, Alison]
slug: stringr-1-4-0
date: '2019-02-21'
description: >
    stringr 1.4.0 is now on CRAN!
categories:
  - learn
photo:
  url: https://unsplash.com/photos/o-d37kiKqqc
  author: Maranda Vandergriff
---



<html>
<style>
h2 code {
    font-size: 1em;
    
}
</style>
</html>

We are happy to announce that [stringr](http://stringr.tidyverse.org/) 1.4.0
is now on CRAN. stringr provides a cohesive set of functions designed to make
working with strings as easy as possible. For a full list of changes, please see the [release notes](https://stringr.tidyverse.org/news/index.html#stringr-1-4-0).

You can install the released version from CRAN:


```r
install.packages("stringr")
```


```r
library(stringr)
```


## New functions

Thanks to the hard work of [John Harmon](https://github.com/jonthegeek) at [Tidyverse Developer Day](https://github.com/tidyverse/dev-day-2019), stringr has three new functions.

[`str_starts()`](https://stringr.tidyverse.org/reference/str_starts.html) and [`str_ends()`](https://stringr.tidyverse.org/reference/str_starts.html) detect the presence or absence of patterns at the beginning or end of strings.
 

```r
fruit <- c("apple", "banana", "pear", "pineapple")
str_starts(fruit, "p")
#> [1] FALSE FALSE  TRUE  TRUE
str_starts(fruit, "p", negate = TRUE)
#> [1]  TRUE  TRUE FALSE FALSE
str_ends(fruit, "e")
#> [1]  TRUE FALSE FALSE  TRUE
str_ends(fruit, "e", negate = TRUE)
#> [1] FALSE  TRUE  TRUE FALSE
```
 
The new [`str_to_sentence()`](https://stringr.tidyverse.org/reference/case.html) function capitalizes strings with sentence case, like so:


```r
str_to_sentence("the quick brown dog")
#> [1] "The quick brown dog"
```

## Support for `negate`

[`str_subset()`](https://stringr.tidyverse.org/reference/str_subset.html), [`str_detect()`](https://stringr.tidyverse.org/reference/str_detect.html), and [`str_which()`](https://stringr.tidyverse.org/reference/str_subset.html) now have the `negate` argument, which is used to find the elements that do _not_ match a pattern (as seen above in the `str_starts()` and `str_ends()` examples).  

## Acknowledgements

Thank you to everyone who contributed to this release: [&#x0040;AmeliaMN](https://github.com/AmeliaMN), [&#x0040;batpigandme](https://github.com/batpigandme), [&#x0040;beckymaust](https://github.com/beckymaust), [&#x0040;BenjaminLouis](https://github.com/BenjaminLouis), [&#x0040;blablablerg](https://github.com/blablablerg), [&#x0040;bschneidr](https://github.com/bschneidr), [&#x0040;bwiernik](https://github.com/bwiernik), [&#x0040;ctmann](https://github.com/ctmann), [&#x0040;damianooldoni](https://github.com/damianooldoni), [&#x0040;dan-reznik](https://github.com/dan-reznik), [&#x0040;denrou](https://github.com/denrou), [&#x0040;diegogarcilazo](https://github.com/diegogarcilazo), [&#x0040;DieselAnalytics](https://github.com/DieselAnalytics), [&#x0040;elisakreiss](https://github.com/elisakreiss), [&#x0040;giovannikraushaar](https://github.com/giovannikraushaar), [&#x0040;hadley](https://github.com/hadley), [&#x0040;hammer](https://github.com/hammer), [&#x0040;jennybc](https://github.com/jennybc), [&#x0040;jimhester](https://github.com/jimhester), [&#x0040;jonocarroll](https://github.com/jonocarroll), [&#x0040;jonthegeek](https://github.com/jonthegeek), [&#x0040;jrnold](https://github.com/jrnold), [&#x0040;juanrocha](https://github.com/juanrocha), [&#x0040;kmace](https://github.com/kmace), [&#x0040;krlmlr](https://github.com/krlmlr), [&#x0040;osorensen](https://github.com/osorensen), [&#x0040;paleolimbot](https://github.com/paleolimbot), [&#x0040;pdelboca](https://github.com/pdelboca), [&#x0040;pgrandinetti](https://github.com/pgrandinetti), [&#x0040;PirateGrunt](https://github.com/PirateGrunt), [&#x0040;samhinshaw](https://github.com/samhinshaw), [&#x0040;sastoudt](https://github.com/sastoudt), [&#x0040;seanpor](https://github.com/seanpor), [&#x0040;yj-danielyang](https://github.com/yj-danielyang), and [&#x0040;yutannihilation](https://github.com/yutannihilation).

