# Contributing to torch

Many thanks for wanting to contribute to `torch`! We welcome all kinds
of contributions, from correcting typos via bug fixes to feature
additions.

## Fixing typos

To fix typos, spelling mistakes, or grammatical errors, you need to make
changes in the *source* `.R` file in question, not in the generated
`.Rd` file. Thereafter, run `tools/document.R` to regenerate the
documentation. (See [Workflow](#workflow) below for why.)

## Filing bugs

If you find a bug in `torch`, please open an issue
[here](https://github.com/mlverse/torch/issues). Please provide detailed
information on how to reproduce the bug. It would be great to also
provide a [`reprex`](https://reprex.tidyverse.org/).

## Feature requests

Feel free to open issues
[here](https://github.com/mlverse/torch/issues), and add the
`feature-request` tag. Also try searching if there’s already an open
issue for your `feature-request` – in this case it’s better to comment
or upvote it instead of opening a new one.

## Examples

We welcome contributed examples. Feel free to open a PR with new
examples. The examples should be placed in the `vignettes/examples`
folder.

The examples should consist of an `.R` file and an `.Rmd` with the same
name that just renders the code.

See
[mnist-mlp.R](https://github.com/mlverse/torch/blob/master/vignettes/examples/mnist-mlp.R)
and
[mnist-mlp.Rmd](https://github.com/mlverse/torch/blob/master/vignettes/examples/mnist-mlp.Rmd)
for an example.

It’s important that one can run the example without manually downloading
any dataset/file. You should also add an entry to the [`_pkgdown.yaml`
file](https://github.com/mlverse/torch/blob/master/_pkgdown.yml#L24-L25)
for your example.

## Code contributions

We have many open issues in the [github
repo](https://github.com/mlverse/torch/issues). If there’s one item that
you want to work on, you can comment on it and ask for directions.

### Code style

- New R code should follow the [`tidyverse` style
  guide](https://style.tidyverse.org). You can use the
  [styler](https://CRAN.R-project.org/package=styler) package to apply
  these styles, or simply run the `tools/style.sh` script, which also
  formats the code and removes whitespace. Please don’t re-style code
  that has nothing to do with your PR.

- New C/C++ code should follow the [`Google` style
  guide](https://google.github.io/styleguide/cppguide.html). You can use
  [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html) to
  apply these styles, or simply run `tools/style.sh`, which also formats
  the code and removes whitespace. Please don’t re-style code that has
  nothing to do with your PR.

- We use [roxygen2](https://cran.r-project.org/package=roxygen2), with
  [Markdown
  syntax](https://cran.r-project.org/web/packages/roxygen2/vignettes/rd-formatting.html),
  to build all documentation for the package.

- We use [testthat](https://cran.r-project.org/package=testthat) for
  unit tests. Contributions with test cases included are easier to
  accept.

### Requirements

- R installation
- R Tools for compilation (only on Windows)
- The `devtools` package
- CMake to compile `lantern` binaries

### Workflow

We use `devtools` as the toolchain for development, but a few steps must
be done before upfront.

The first time you clone the repository, run:

``` r
source("tools/buildlantern.R")
```

This will first download LibTorch and copy its binaries to the `deps`
folder in the working directory, and then, compile the `lantern`
binaries themselves.

This command must be run everytime you modify `lantern` code, i.e., code
that lives in `lantern/src`.

You can then run

``` r
devtools::load_all()
```

to load `torch`, and test interactively. Alternatively, run

``` r
devtools::test()
```

to execute the test suite.

Finally, it’s important to update the documentation. Please always use
the custom `tools/document.R` script instead of `devtools::document()`,
as we need to patch `roxygen2` to avoid running the examples on CRAN.

We’re looking forward to your contributions!

### Known Issues

- When running the tests twice (e.g. via `devtools::test()`) this will
  cause a segfault, see
  [\#1218](https://github.com/mlverse/torch/issues/1218) for a
  discussion.
