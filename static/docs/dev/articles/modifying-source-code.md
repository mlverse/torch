# Modifying torch source code

## Overview

The torch R package is composed of multiple abstraction layers until R
code finally calls LibTorch code. Depending on which part of the code
you want to modify you might need different build requirements for
setting up.

Because of some special requirements the project has, it’s not always
possible to use the standard `devtools` tools to work on torch. This
guide is intended for contributors to the torch codebase to quickly get
set up and be able to modify the source code without spending too much
time understanding the internals.

The torch R package is composed of the following layers, going top down:

1.  R code calling `torch_` functions.
2.  R code calling `cpp_torch_` functions.
3.  Rcpp (C++) code calling `lantern_torch_` functions.
4.  Lantern (C++) code calling LibTorch functions.

Depending on the function 2, 3 and 4. might be auto-generated (via the
tools/torchgen) package or not. In the next sections we will explain how
to contribute code depending on the layer you are modifying.

## R source code

Modifying R source code should be the simplest case and it shouldn’t be
required for contributors to other tools than the standard R development
toolbox (like RTools on Windows and a C++ compiler on MacOS).

The standard `devtools` should just work too. Ie, you can clone the
project from GitHub, modify R source code, including documentation and
run `devtools::load_all()`

The first time you run `devtools::load_all()`, the torch installation
prompt will appear and LibTorch and lantern binaries will be downloaded
and installed inside the `inst` directory of your installation.
Subsequent calls will work as expected.

> **Note** loading torch twice without restarting the R session causes
> the session to **crash**. Unbfortunatelly we don’t know yet how to fix
> this bug and it’s related to problems with correctly unloading shared
> libraries. So you should **restart** your session before calling
> `devtools::load_all()` again. Also **note** that other functions might
> call `devtools::load_all()` internally, like eg. `devtools::test()`
> and `devtools::document()`.

Also, running **Build** or **Check** in the RStudio pane will fail, as
it calls Rcpp::CompileAttributes without calling the `Makevars` file
instructions, that are required for torch to work correctly.

If you use `devtools::check()` pass the argument
`devtools::check(document=FALSE)`. The reason for this is described
below.

### Documentation

torch registers a few roxygen2 roclets to handle examples and sessions
in roxygen2 documentation. Unfortunately we are unaware of any method to
automatically register these roclets without messing with the
DESCRIPTION’s collate field.

The recommended way to update documentation in torch is by using the
`tools/document.R` script. You can either call
`source("tools/document.R")` or, the recommended, run:

    Rscript tools/document.R

in your terminal. This has the advantage that since
`devtools::document()` is called internally and it calls
[`pkgload::load_all()`](https://pkgload.r-lib.org/reference/load_all.html),
that won’t crash your R session if you forget to restart your session
(see note above).

## Rcpp code

Modifying Rcpp source code in `src` should be very similar to modifying
C++ in other R packages.

You should pay attention when calling `lantern_*` functions though. Most
`lantern_*` functions return `void*` that will leak the referring
objects if they are not correctly freed after use. The best way to
handle them is to always assign them to a variable with the correct type
defined in `inst/include/torch_types.h` specially those defined in the
C++ `namespace torch {}`. If you use these kind of objects you also gain
the ability to return them to R without having to write custom Rcpp code
as for most of them we have implemented the `SEXP()` operator.

## Lantern code

In order to modify the Lantern source code you will first need to make
sure the environment variable `BUILD_LANTERN` is set to `1`. To avoid
forgetting to set it you can add it to your `.Renviron` using, eg
`usethis::edit_r_environ()`.

That flag will trigger the `configure` file to call the `lantern` target
in Makevars. Building lantern requires CMake to be in your path. You can
install CMake in all major platforms by following instructions in the
[install page](https://cmake.org/install/).

When `BUILD_LANTERN=1` and you run `devtools::load_all()`, Lantern will
be compiled as part of that workflow. You will see a new directory
called `build-lantern` in your package directory containing files
related to the Lantern build.

Lantern source code is located in `src/lantern/src`. The lantern
CMakeLists file is located at `src/lantern/CMakeLists.txt`. You can also
take a look at the `lantern` target in `src/Makevars.in` to get a sense
of what CMake commands are called when building Lantern from source.

By default, rebuilding Lantern is incremental, ie, calling
`devtools::load_all()` will only rebuild modified parts of the Lantern
code. Depending on the extend of your modifications in Lantern source
code you might need to completely clean the build directory, you can do
it by simply removing the `build-lantern` directory.
