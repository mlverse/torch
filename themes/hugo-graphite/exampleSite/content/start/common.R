knitr::opts_chunk$set(
  digits = 3,
  comment = "#>",
  dev = 'svg', 
  dev.args = list(bg = "transparent"),
  fig.path = "figs/",
  collapse = TRUE,
  cache.path = "cache/"
)
options(width = 80)

req_pkgs <- function(x, what = "To use code in this article, ") {
  x <- sort(x)
  x <- knitr::combine_words(x, and = " and ")
  paste0(
    what,
    " you will need to install the following packages: ",
    x, "." 
  )
}
small_session <- function(pkgs = NULL) {
  pkgs <- c(pkgs, "recipes", "parsnip", "tune", "workflows", "dials", "dplyr",
            "broom", "ggplot2", "purrr", "rlang", "rsample", "tibble", "infer",
            "yardstick", "tidymodels", "infer")
  pkgs <- unique(pkgs)
  library(sessioninfo)
  library(dplyr)
  sinfo <- sessioninfo::session_info()
  cls <- class(sinfo$packages)
  sinfo$packages <- 
    sinfo$packages %>% 
    dplyr::filter(package %in% pkgs)
  class(sinfo$packages) <- cls
  sinfo
}
