---
title: roxygen2 7.1.0
author: Gábor Csárdi
date: '2020-03-11'
slug: roxygen2-7-1-0
categories:
  - package
description: >
  A minor roxygen2 release with some new features
tags:
  - r-lib
  - devtools
  - roxygen2
photo:
  url: https://pixabay.com/photos/fabric-wool-textile-texture-1031932/
  author: John Perrett
---

We're chuffed to announce the release of [roxygen2 7.1.0](https://roxygen2.r-lib.org).
roxygen2 allows you to write specially formatted R comments that generate R documentation files (`man/*.Rd`) and the `NAMESPACE` file.
roxygen2 is used by over 8,800 CRAN packages.

Install the latest version of roxygen2 with:


```r
install.packages("roxygen2")
```

This is a minor release, with many small improvements and bug fixes, and some new features.
This blog post discusses the most important changes. See the [full changelog](https://roxygen2.r-lib.org/news/index.html#roxygen2-7-1-0) for all changes.

## roxygen2 meets knitr

We have been gradually improving roxygen2's markdown support, and this release has an exciting new markdown feature as well.
Starting from roxygen2 7.1.0 you can write inline R code and markdown code fences directly in roxygen2 comments!
roxygen2 runs the R code and inserts its output into the documentation.

### Inline R code

For running R code inline, use the usual markdown backticks, and prefix the code with `r` and a space character.
roxygen2 interprets the rest of the text within backticks as R code, evaluates it, and replaces the backtick expression with its value.
After all such substitutions, the text of the whole tag is interpreted as markdown, as usual.

For example, the following code inserts the date and the R version of the roxygen2 run.

```r
#' roxygen2 created this manual page on `r Sys.Date()` using R version
#' `r getRversion()`.
```

### Code blocks (fences)

Markdown code blocks can be dynamic as well, if you use <code>```{r}</code> to start them, just like in knitr documents.
roxygen evaluates the code, and (by default) both the code and the printed value of the expression will be inserted into the manual page.

```r
#' ```{r}
#' # This block of code will be evaluated
#' summary(iris)
#' ```
```

Code blocks support knitr chunk options, e.g. to keep the output of several expressions together, you can specify `results= "hold"`:

```r
#' ```{r results = "hold"}
#' names(mtcars)
#' nrow(mtcars)
#' ```
```

By default plots create `.png` files in the `man/figures` directory. The file names are created from the chunk names:

```r
#' ```{r r-cookbook-barplot}
#' # https://r-graphics.org/recipe-distribution-basic-hist
#' library(ggplot2)
#' ggplot(faithful, aes(x = waiting)) +
#'   geom_histogram()
#' ```
```

Both the inline R code and the markdown code fences are evaluated when you run `devtools::document()` (or `roxygenize()`).
You probably want to avoid lengthy computations, or set up caching to keep your package development workflow snappy.

Please see `vignette("rd-formatting")` for more about dynamic documentation, including potential caveats.

## Line ending conversion

This version of roxygen2 does a much better job at keeping the line ending characters consistent within each file, across operating systems.
In particular, if a generated file has only Windows (CR LF) line endings, roxygen2 keeps the file that way.
If a file has mixed Windows and Unix (LF) line endings, roxygen2 converts all line endings to Unix (LF).
For new files roxygen2 uses LF file endings.
If you want to convert a file from CR LF line endings to LF, simply remove it and let roxygen2 re-create it.

The new convention works better with git on Windows.
Windows git, depending on configuration, might or might not convert between CR LF and LF line endings when checking out and when committing code.
No matter which git option you use, roxygen2 now does not introduce spurious line ending changes.

Thanks to [&#x0040;jonthegeek](https://github.com/jonthegeek) and [&#x0040;jimhester](https://github.com/jimhester) for working on this feature.

## Some other improvements

* Hyperlinks to R6 methods are also added in the PDF manual.

* `@description NULL` and `@details NULL` no longer fail; instead, these tags
  are ignored, except for `@description NULL` in package level documentation,
  where it can be used to suppress the auto-generated 'Description' section.

* Multiple `@format` tags are now combined.

* `@evalNamespace()` works again.

## Acknowledgements

A big thanks to all 40 contributors who helped make this release possible! [&#x0040;alandipert](https://github.com/alandipert), [&#x0040;allenzhuaz](https://github.com/allenzhuaz), [&#x0040;BenEngbers](https://github.com/BenEngbers), [&#x0040;bgctw](https://github.com/bgctw), [&#x0040;billdenney](https://github.com/billdenney), [&#x0040;Bisaloo](https://github.com/Bisaloo), [&#x0040;cboettig](https://github.com/cboettig), [&#x0040;dmurdoch](https://github.com/dmurdoch), [&#x0040;dragosmg](https://github.com/dragosmg), [&#x0040;eddelbuettel](https://github.com/eddelbuettel), [&#x0040;gaborcsardi](https://github.com/gaborcsardi), [&#x0040;genomaths](https://github.com/genomaths), [&#x0040;goldingn](https://github.com/goldingn), [&#x0040;hadley](https://github.com/hadley), [&#x0040;HenrikBengtsson](https://github.com/HenrikBengtsson), [&#x0040;Hong-Revo](https://github.com/Hong-Revo), [&#x0040;hughjonesd](https://github.com/hughjonesd), [&#x0040;iferres](https://github.com/iferres), [&#x0040;IndrajeetPatil](https://github.com/IndrajeetPatil), [&#x0040;jameslamb](https://github.com/jameslamb), [&#x0040;jimhester](https://github.com/jimhester), [&#x0040;kingaa](https://github.com/kingaa), [&#x0040;kortschak](https://github.com/kortschak), [&#x0040;krlmlr](https://github.com/krlmlr), [&#x0040;maelle](https://github.com/maelle), [&#x0040;michaelquinn32](https://github.com/michaelquinn32), [&#x0040;mikemahoney218](https://github.com/mikemahoney218), [&#x0040;mstr3336](https://github.com/mstr3336), [&#x0040;muschellij2](https://github.com/muschellij2), [&#x0040;nteetor](https://github.com/nteetor), [&#x0040;ntguardian](https://github.com/ntguardian), [&#x0040;pat-s](https://github.com/pat-s), [&#x0040;RaphaelS1](https://github.com/RaphaelS1), [&#x0040;russHyde](https://github.com/russHyde), [&#x0040;s-fleck](https://github.com/s-fleck), [&#x0040;stefanfritsch](https://github.com/stefanfritsch), [&#x0040;strboul](https://github.com/strboul), [&#x0040;TomKellyGenetics](https://github.com/TomKellyGenetics), [&#x0040;VPetukhov](https://github.com/VPetukhov), and [&#x0040;zachary-foster](https://github.com/zachary-foster).
