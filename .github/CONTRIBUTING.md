# Contributing to torch

This outlines how to propose a change to torch. 
For more detailed info about contributing to this, and other tidyverse packages, please see the
[**development contributing guide**](https://rstd.io/tidy-contrib). 

## Fixing typos

You can fix typos, spelling mistakes, or grammatical errors in the documentation directly using the GitHub web interface, as long as the changes are made in the _source_ file. 
This generally means you'll need to edit [roxygen2 comments](https://roxygen2.r-lib.org/articles/roxygen2.html) in an `.R`, not a `.Rd` file. 
You can find the `.R` file that generates the `.Rd` by reading the comment in the first line.

See also the [Documentation] section.

## Filing bugs

If you find a bug in `torch` please open an issue [here](https://github.com/mlverse/torch/issues).
Please, provide detailed information on how to reproduce the bug. It would be great
to also provide a [`reprex`](https://reprex.tidyverse.org/).

## Feature requests

Feel free to open issues [here](https://github.com/mlverse/torch/issues) and add the
`feature-request` tag. Try searching if there's already an open issue for your
`feature-request`, in this case it's better to comment or upvote it intead of opening
a new one.

## Examples

We welcome contributed examples. feel free to open a PR with new examples.
The should be placed in the `vignettes/examples` folder.

The examples should be an .R file and a .Rmd file with the same name that
just renders the code.

See [mnist-mlp.R](https://github.com/mlverse/torch/blob/master/vignettes/examples/mnist-mlp.R) and
[mnist-mlp.Rmd](https://github.com/mlverse/torch/blob/master/vignettes/examples/mnist-mlp.Rmd)

One must be able to run the example without manually downloading any dataset/file.
You should also add an entry to the [`_pkgdown.yaml` file](https://github.com/mlverse/torch/blob/master/_pkgdown.yml#L24-L25).

## Code contributions

We have many open issues in the [github repo](https://github.com/mlverse/torch/issues)
if there's one item that you want to work on, you can comment on it an ask for
directions.

## Documentation

We use roxygen2 to generate the documentation. IN order to update the docs, edit
the file in the `R` directory. To regenerate and preview the docs, use the custom
`tools/document.R` script, as we need to patch roxygen2 to avoid running the examples
on CRAN.
