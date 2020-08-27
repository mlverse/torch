---
title: Tidyverse packages
---

## Installation and use

* Install all the packages in the tidyverse by running `install.packages("tidyverse")`.

* Run `library(tidyverse)` to load the core tidyverse and make it available
  in your current R session.

Learn more about the tidyverse package at <https://tidyverse.tidyverse.org>.

## Core tidyverse

The core tidyverse includes the packages that you're likely to use in everyday data analyses. As of tidyverse 1.3.0, the following packages are included in the core tidyverse:

{{< packages-hex >}}

The tidyverse also includes many other packages with more specialised usage. They are not loaded automatically with `library(tidyverse)`, so you'll need to load each one with its own call to `library()`.

## Import

As well as [readr](https://readr.tidyverse.org), for reading flat files, the tidyverse package installs a number of other packages for reading data:

* [DBI](https://github.com/rstats-db/DBI) for relational databases.
  (Maintained by [Kirill Müller](https://www.cynkra.com).)
  You'll need to pair DBI with a database specific backends like 
  [RSQLite](https://rsqlite.r-dbi.org), 
  [RMariaDB](https://rmariadb.r-dbi.org),
  [RPostgres](https://rpostgres.r-dbi.org), or 
  [odbc](https://github.com/r-dbi/odbc). 
  Learn more at <https://db.rstudio.com>.

* [haven](https://haven.tidyverse.org) for SPSS, Stata, and SAS data.

* [httr](https://github.com/r-lib/httr) for web APIs.

* [readxl](https://readxl.tidyverse.org) for `.xls` and `.xlsx` sheets.

* [rvest](https://github.com/tidyverse/rvest) for web scraping.

* [jsonlite](https://github.com/jeroen/jsonlite#jsonlite)
  for JSON. (Maintained by [Jeroen Ooms](https://github.com/jeroen).)

* [xml2](https://github.com/r-lib/xml2) for XML.

## Wrangle

In addition to [tidyr](https://tidyr.tidyverse.org), and [dplyr](https://dplyr.tidyverse.org), there are five packages (including [stringr](https://stringr.tidyverse.org) and [forcats](https://forcats.tidyverse.org)) which are designed to work with specific types of data:

* [lubridate](https://lubridate.tidyverse.org) for dates and date-times.
* [hms](https://github.com/tidyverse/hms) for time-of-day values.
* [blob](https://github.com/tidyverse/blob) for storing blob (binary) data.

## Program

In addition to [purrr](https://purrr.tidyverse.org), which provides very consistent and natural methods for iterating on R objects, there are two additional tidyverse packages that help with general programming challenges:

* [magrittr](https://magrittr.tidyverse.org) provides the pipe, `%>%` used
  throughout the tidyverse. It also provide a number of more specialised
  piping operators (like `%$%` and `%<>%`) that can be useful in other places.

* [glue](https://github.com/tidyverse/glue) provides an alternative to
  `paste()` that makes it easier to combine data and strings.

## Model

Modeling with the tidyverse uses the collection of [tidymodels packages](https://www.tidymodels.org/), which largely replace the [modelr](https://github.com/tidyverse/modelr) package used in [R4DS](https://r4ds.had.co.nz/). These packages provide a comprehensive foundation for creating and using models of all types. Visit the [_Getting Started_](https://www.tidymodels.org/start/) guide or, for more detailed examples, go straight to the [_Learn_](https://www.tidymodels.org/learn/) page.  

## Get help

If you’re asking for R help, reporting a bug, or requesting a new feature, you’re more likely to succeed if you include a good reproducible example, which is precisely what the [reprex](https://reprex.tidyverse.org/) package is meant for. You can learn more about reprex, along with other tips on how to help others help you in the [help section](https://www.tidyverse.org/help/).

