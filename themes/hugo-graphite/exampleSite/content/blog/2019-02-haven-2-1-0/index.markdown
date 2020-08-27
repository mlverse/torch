---
title: haven 2.1.0
author: 
  - carl
  - Garrett
slug: haven-2-1-0
date: '2019-02-19'
description: > 
  haven 2.1.0 is now on CRAN!
categories:
  - other
photo:
  url: https://www.pexels.com/photo/beach-lighthouse-3491/
  author: Skitterphoto
---



We're delighted to announce that [haven 2.1.0](https://haven.tidyverse.org/) is now on CRAN. haven enables R to read and write various data formats used by other statistical packages by wrapping the [ReadStat](https://github.com/WizardMac/ReadStat) C library written by [Evan Miller](https://www.evanmiller.org/). For a full account of updates in this release, see the [Changelog](https://haven.tidyverse.org/news/index.html).

## Improved labelling 

Both [`labelled()`](https://haven.tidyverse.org/reference/labelled.html) and [`labelled_spss()`](https://haven.tidyverse.org/reference/labelled_spss.html) now allow `NULL` labels. This makes both classes more flexible, allowing you to use them for their other attributes.`labelled()` also now tests that value labels are unique.


`labelled` objects now get pretty printing that shows the labels and `NA` values when inside of a `tbl_df`. You can turn this behaviour off by using `option(haven.show_pillar_labels = FALSE)`. 


```r
tibble::tibble(s = haven::labelled(c(1, 10), labels = c("A" = 1, "B" = 10)))
#> # A tibble: 2 x 1
#>   s        
#>   <dbl+lbl>
#> 1  1       
#> 2 10
```


## Minor improvements and fixes

This release is updated to the latest version of Evan Miller's [ReadStat](https://github.com/WizardMac/ReadStat), which includes the following changes:

 * [`read_por()`](https://haven.tidyverse.org/reference/read_spss.html) can now read files from SPSS 25.  
 * [`read_por()`](https://haven.tidyverse.org/reference/read_spss.html) uses base-10 instead of base-30 for the exponent.  
 * [`read_sas()`](https://haven.tidyverse.org/reference/read_sas.html) can read zero-column files.  
 * [`read_sav()`](https://haven.tidyverse.org/reference/read_spss.html) now reads long strings, and has greater memory limit, allowing it to read more labels.  
 * [`read_spss()`](https://haven.tidyverse.org/reference/read_spss.html) reads long variable labels.  
 * [`write_sav()`](https://haven.tidyverse.org/reference/read_spss.html) no longer creates incorrect column names when >10k columns.
 * [`write_sav()`](https://haven.tidyverse.org/reference/read_spss.html) no longer crashes when writing long label names.  

## Acknowledgements

Thank you to Evan Miller, as well as 
[&#x0040;armenic](https://github.com/armenic),  [&#x0040;beckerbenj](https://github.com/beckerbenj), [&#x0040;caayala](https://github.com/caayala), [&#x0040;gergness](https://github.com/gergness), [&#x0040;jeffeaton](https://github.com/jeffeaton),  [&#x0040;philstraforelli](https://github.com/philstraforelli), [&#x0040;thays42](https://github.com/thays42), and [&#x0040;visseho](https://github.com/visseho) for their contributions to this release.

