---
title: Self-cleaning test fixtures
author: Jenny Bryan
date: '2020-04-27'
slug: self-cleaning-test-fixtures
categories:
  - programming
  - learn
tags:
  - r-lib
description: |
  A wild romp through environments -- namely, the environments associated with
  functions and tests. How to adopt a low-impact lifestyle.
photo:
  url: https://twitter.com/MariannaFoos
  author: Marianna Foos
---



*Adapted from an internal presentation to the tidyverse team, which tells you something about the target reader. The primary audience for this post is R programmers and, especially, package developers. The problems identified and solved here are pretty niche! People who are mostly interested in R as a data analysis tool may not have direct use for this material. But the post offers something for anyone curious about the hazards of side effects and the various ways we can leave the world as you found it.*

## Test hygiene

> Take nothing but memories, leave nothing but footprints.

― Chief Si'ahl

Ideally a test should leave the world exactly as it found it. Examples of things you might do inside a test and, therefore, need to undo:

* Create a file or directory
* Create a resource on an external system
* Set an R option
* Set an environment variable
* Change working directory
* Change an aspect of the tested package's state

Scrupulous attention to cleanup is more than just courtesy or being fastidious. It is also self-serving. The state of the world after test `i` is the starting state for test `i + 1`. Tests that change state willy-nilly eventually end up interfering with each other in ways that can be very difficult to debug. Most tests are written with an implicit assumption about the starting state, usually whatever *tabula rasa* means for the target domain of your package. If you accumulate enough sloppy tests, you will eventually find yourself asking the programming equivalent of questions like "Who forgot to turn off the oven?" and "Who didn't clean up after the dog?".

First, we lay some foundations that aren't obviously related to tests, but just trust that we'll get there eventually.

##  The `on.exit()` pattern

If you want to clean up after yourself, how should you actually do it?

The first function to know about is base R's [`on.exit()`](https://rdrr.io/r/base/on.exit.html). You use it inside a function. In the function body, every time you do something that should be undone **on exit**, you immediately register the cleanup code with `on.exit(expr, add = TRUE)`[^on-exit-add].

[^on-exit-add]: It's too bad `add = TRUE` isn't the default, because you almost always want this. Without it, each call to `on.exit()` clobbers the effect of previous calls.

Here's a `sloppy()` function that prints a number with a specific number of significant digits, by adjusting an R option.




```r
sloppy <- function(x, sig_digits) {
  options(digits = sig_digits)
  print(x)
}

pi
#> [1] 3.141593

sloppy(pi, 2)
#> [1] 3.1

pi
#> [1] 3.1
```



Notice how `pi` prints differently before and after the call to `sloppy()`. Calling `sloppy()` has a side effect: it changes the `digits` option globally, not just within its own scope of operations. This is what we want to avoid.

*Don't worry, I'm restoring global state (specifically, the `digits` option) behind the scenes here.*

Here's how to do better with `on.exit()`.


```r
neat <- function(x, sig_digits) {
  op <- options(digits = sig_digits)
  on.exit(options(op), add = TRUE)
  print(x)
}

pi
#> [1] 3.141593

neat(pi, 2)
#> [1] 3.1

pi
#> [1] 3.141593
```

The use of `on.exit()` ensures that `neat()` leaves `digits` the way that it found it. `on.exit()` also works when you exit the function abnormally, i.e. due to error. This is why it's a better choice than any do-it-yourself solution.

But I promised to talk about tests! Never fear, `on.exit()` also works inside a test.


```r
library(testthat)

exp(1)
#> [1] 2.718282

test_that("on.exit() works in a test", {
  op <- options(digits = 2)
  on.exit(options(op), add = TRUE)
  out <- format(exp(1))
  expect_equal(out, "2.7")
  # printing just for the benefit of the reader
  print(out) 
})
#> [1] "2.7"

exp(1)
#> [1] 2.718282
```

`on.exit()` is a very useful function and provides enough inspiration for an entire package: withr ([withr.r-lib.org](http://withr.r-lib.org)), which is a Swiss army knife for managing state in very flexible ways. It's what I usually use, in functions and tests, for situations like that above.

*For more background, the section about [Exit handlers](https://adv-r.hadley.nz/functions.html#on-exit) in Advanced R is a good reference. The [cleancall package](https://github.com/r-lib/cleancall#readme) addresses a similar problem, but in the C code of an R package. cleancall is introduced in the blog post [Resource Cleanup in C and the R API](https://www.tidyverse.org/blog/2019/05/resource-cleanup-in-c-and-the-r-api/).*

## `withr::defer()`

`withr::defer()` is a more general version of `on.exit()`. It can run cleanup for any environment, but defaults to the environment it was called in. Therefore, it works like `on.exit()` inside a function -- an extremely important special case -- but the added flexibility means you can use it in more situations.

Below I compare `on.exit()` and `withr::defer()` and I put the code inside `local()`, instead of inside a function. This is meant to reinforce that cleanup can be relevant beyond function execution environments. It also gives you another tool to play with, in addition to toy functions and tests, in your own explorations of how to manage scope.


```r
library(withr)

local({
  on.exit(print("first"))
  on.exit(print("last"))  # this clobbers `print("first")` :(
})
#> [1] "last"

local({
  on.exit(print("first"), add = TRUE)
  on.exit(print("last"), add = TRUE)
})
#> [1] "first"
#> [1] "last"

local({
  on.exit(print("first"), add = TRUE, after = FALSE)
  on.exit(print("last"), add = TRUE, after = FALSE)
})
#> [1] "last"
#> [1] "first"

local({
  defer(print("first"))
  defer(print("last"))
})
#> [1] "last"
#> [1] "first"
```

This showcases the nice ergonomics of `defer()`: each call *adds* to the list of deferred tasks (vs. replaces) and, by default, adds to the *front* of the stack (vs. the back). As you'll see below, this turns out to matter in most real world usage[^on-exit-after].

[^on-exit-after]: Note: the `after` argument of `on.exit()` first appeared in R 3.5.0.

## `withr::local_*()`

`on.exit()` usage has a very predictable, slightly clunky pattern. In `neat()`, it looks like:


```r
op <- options(digits = sig_digits)
on.exit(options(op), add = TRUE)      
```

The first statement accomplishes two things at once: it sets the `digits` option and captures its original value in `op`. The second statement schedules the restoration of `digits`. This order of operations is encouraged by the design of `options()`, which returns the original values when it's used as a setter.

Here is the more general (and possibly slightly safer) approach: First, capture the current state. Then, immediately schedule the eventual restoration of this original state, so that this is arranged before any additional calls are made that could exit, e.g. throw an error. Last, make the desired state change.

What if such a maneuver happens all over your package and you want to write a helper?

You can't wrap `on.exit()` in your own helpers, because there's no way to reach back up into the correct parent frame and schedule cleanup there. But with `defer()`, we can! Here is such a custom helper, called `local_digits()`.


```r
local_digits <- function(sig_digits, env = parent.frame()) {
  op <- options(digits = sig_digits)
  defer(options(op), env = env)
}
```

We can use it to keep any manipulation of `digits` local to a test (or function).


```r
exp(1)
#> [1] 2.718282

test_that("withr lets us write custom helpers for local state manipulation", {
  local_digits(20)
  print(exp(1))
})
#> [1] 2.7182818284590450908

exp(1)
#> [1] 2.718282

test_that("we can even make multiple calls to local_digits()", {
  local_digits(1)
  print(exp(1))
  local_digits(3)
  print(exp(1))
  local_digits(5)
  print(exp(1))
})
#> [1] 3
#> [1] 2.72
#> [1] 2.7183

exp(1)
#> [1] 2.718282
```

The ability to write `on.exit()`-like functions, customized to your own needs, is very empowering. However, you may not even need to. There are certain state changes that come up over and over again, for all of us. These are pre-implemented in withr's `local_*()` family of functions. A few examples:

| Do / undo this              | withr function    |
|-----------------------------|-------------------|
| Create a file               | `local_file()`    |
| Set an R option             | `local_options()` |
| Set an environment variable | `local_envvar()`  |
| Change working directory    | `local_dir()`     |

"Local" here refers to the fact that the state change persists only for the lifetime of an associated environment, usually the execution environment of a function or test.

We can use `withr::local_options()` to write yet another version of `neat()`:


```r
neater <- function(x, sig_digits) {
  local_options(list(digits = sig_digits))
  print(x)
}

pi
#> [1] 3.141593

neater(pi, 3)
#> [1] 3.14

pi
#> [1] 3.141593
```

Each `local_*()` function has a companion `with_()` function, which is a nod to [`with()`](https://rdrr.io/r/base/with.html). We won't use the `with_*()` functions here, but you can learn more about them at [withr.r-lib.org](http://withr.r-lib.org).

## Test fixtures

Testing is often demonstrated with cute little tests and functions where all the inputs and expected results can be inlined. But in real packages, things aren't always so simple. The main functions in your package probably aren't "1 number in, 1 number out". They might require more exotic objects or very specific circumstances. Changing state might be the entire purpose of a function! Now what?

*Obligatory caveat: If you find it hard to write tests, this may be the universe telling you that your package has some design problems. Maybe you've somehow ended up with a small number of monster functions, with oodles of arguments and complex conditional logic, that can do everything from scramble eggs to change a lightbulb. The best move in this case may be to break things up into smaller and simpler functions. And those will be easier to test. End caveat.*

Tricky test situations can't always be eliminated by better package design. Let's assume you've got a reasonable design and you're still facing some test dilemmas. Unless you find a way to make writing tests as pleasant as possible, you won't write nearly enough of them.

One technique I've found useful is what I'll call a **self-cleaning test fixture**.

### usethis and `create_local_package()`

The usethis package ([usethis.r-lib.org](https://usethis.r-lib.org)) provides functions for looking after the files and folders in an R project, especially an R package. These function names should give you a vague sense of what usethis does: `create_package()`, `use_vignette()`, `use_testthat()`, `use_github()`. Many of these functions only make sense in the context of an R package. That means in order to test them, we have to be working inside an R package. And they can't all target some persistent Frankenpackage.

We need a way to quickly spin up a minimal package, in the session temp directory. Test some functions against it. Then destroy it. Without a lot of fuss. We need a **local package**.


```r
create_local_package <- function(dir = file_temp(), env = parent.frame()) {
  old_project <- proj_get_()            # --- Record starting state --- 
  
  withr::defer({                        # --- Defer The Undoing --- 
    proj_set(old_project, force = TRUE) # restore active usethis project (-C)
    setwd(old_project)                  # restore working directory      (-B)
    fs::dir_delete(dir)                 # delete the temporary package   (-A)
  }, envir = env)
                                        # --- Do The Doing ---      
  create_package(dir, open = FALSE)     # create new folder and package  (A)
  setwd(dir)                            # change working directory       (B)
  proj_set(dir)                         # switch to new usethis project  (C)
  invisible(dir)
}
```

That's a simplified version of the test helper[^test-helpers] we use in over 170 tests in usethis. Here's an example of how `create_local_package()` is used in a test:

[^test-helpers]: `create_local_package()` is a test helper. The testthat package allows such things to be defined in `tests/testthat/helper.R` and then makes them available within package tests. They are also loaded by `devtools::load_all()`. `tests/testthat/helper.R` is also a great place to define custom expectations.


```r
test_that("use_roxygen_md() adds DESCRIPTION fields", {
  skip_if_not_installed("roxygen2")
  
  pkg <- create_local_package() # <<<<<------------------------ HERE IT IS!!!!!
  
  use_roxygen_md()
  
  expect_identical(
    desc::desc_get("Roxygen", pkg),
    c(Roxygen = "list(markdown = TRUE)")
  )
  expect_true(desc::desc_has_fields("RoxygenNote", pkg))
  expect_true(uses_roxygen_md())
})
```

This test checks that `usethis::use_roxygen_md()` does the setup necessary to use roxygen2 in a package and, specifically, to write documentation with markdown syntax. All 3 expectations consult the DESCRIPTION file, directly or indirectly. So it's very convenient that `create_local_package()` creates a minimal package, with a valid DESCRIPTION file, for us to test against. And when the test is done -- poof! -- the package is gone.

The setup and teardown done by `create_local_package()` would be aggravating and repetitive to inline in each individual test. The tests would be dominated by this code, making them less readable. If we need to tweak something, we'd have to do it in hundreds of places. This sort of friction has a real chilling effect on one's enthusiasm for writing and maintaining tests.
    
A few more observations about the self-cleaning test fixture pattern:

  * Every action has an equal and opposite reaction. Each individual "doing"
    action (A) has a matching, deferred "undoing" reaction (-A).
  * We work in this order (usually and preferably):
    - Record existing state.
    - Describe the eventual cleanup.
    - Make the desired state change.
  * The undoing usually unfolds in the opposite order from the doing ("last in,
    first out"). This is almost always OK and it is often absolutely necessary.
    In `create_local_package()`:
    - Doing: We must create directory `dir` (A) before we can make it the
      working directory (B). (A) must come before (B).
    - Undoing: We must restore the original working directory (-B) before
      we can delete `dir` (-A). (-B) must come before (-A). We can't delete
      `dir` while it's still the working directory!
    - Think of it like a stack of plates: the last plate onto the stack has to
      be the first one off.
    
**Test fixture** is a pre-existing term in the software engineering world (and beyond):

> A test fixture is something used to consistently test some item, device, or piece of software.

-- [Wikipedia](https://en.wikipedia.org/wiki/Test_fixture)

When I first heard "test fixture" (from Gábor Csárdi, I think), a light bulb clicked on in my head. This was something I *knew* I needed and had even implemented in various half-baked ways. But I hadn't identified it as A Real Thing, with specific goals and design principles. It's a great example of [hypocognition](https://blogs.scientificamerican.com/observations/unknown-unknowns-the-problem-of-hypocognition/). Learning the term "test fixture" gave me a place to hang this knowledge and allowed me to more quickly identify situations where a test fixture was needed.

### googlesheets4 and `local_ss()`

The googlesheets4 package ([googlesheets4.tidyverse.org](https://googlesheets4.tidyverse.org)) provides an R interface to the Google Sheets API. A typical test needs access to a Google Sheet, constructed to have very specific properties and the test may even need to modify the Sheet[^mocking].

[^mocking]: You might ask about mocking here, but I usually don't embrace that heavily and, in any case, that is a topic for another day.

I need a way to quickly create a Sheet, possibly with very specific initial worksheets, cell data, locale, time zone, etc. Test some functions against it. Then trash it. I need a *local spreadsheet*.

Here's a simplified version of the helper `local_ss()`:


```r
local_ss <- function(name, ..., env = parent.frame()) {
  existing <- gs4_find(name)
  if (nrow(existing) > 0) {
    stop_glue("A spreadsheet named {sq(name)} already exists.")
  }

  withr::defer({
    trash_me <- gs4_find(name)
    googledrive::drive_trash(trash_me)
  }, envir = env)
  
  gs4_create(name, ...)
}
```

Even though the Sheets API is very file-ID-oriented, I go out of my way to work here via Sheet name. I bring this up to illustrate another point: you can also use a helper like this to rationalize your development workflow.

At first, it feels like `local_ss()` should create a new Sheet, store its ID, and then schedule it for deletion. But reality is more messy. As I develop a function and its tests, my experimentation can leave behind several instances of a test Sheet (yes, on Drive, you can have several files with the same name!). This leads to a very cluttered and confusing situation in the test account, requiring a periodic "search and destroy" mission for zombie test Sheets. Now my helper immediately alerts me to this problem and applies constant pressure to keep things neat and tidy.

If you keep stubbing your toe in a particular way as you work on your package, zoom out and consider if you can design the problem away by adjusting your workflow. The helper that creates a self-cleaning test fixture is great place to apply this sort of leverage.

## `defer()` on the global environment

I conclude with one more story about workflow. We've talked about two main functions for registering deferred events: base R's `on.exit()` and `withr::defer()`. Part of what `withr::defer()` offers over `on.exit()` is the ability to defer events on *any* environment. But there was still a big exception: the global environment.

Until quite recently, here's what happened if you called `defer()` in an interactive R session[^on-exit-global-env]:

[^on-exit-global-env]: For all practical purposes, you get the same result with `on.exit()`. It's just a silent no-op.


```r
withr::defer(print("hi"))
#> Error in withr::defer(print("hi")):
#>   attempt to defer event on global environment

packageVersion("withr")
#> [1] '2.1.2'
```

Frankly, this makes a lot of sense. Deferred events are triggered when an environment goes out of scope. `on.exit()` and `defer()` are meant to be used in an ephemeral environment, like a function execution environment. Deferring events on the global environment is pretty weird.

But what about your interactive development of functions and tests? Every time you hit a call to `defer()` or `local_*()`, that code fails to execute. You're forced to develop your logic at arm's length or implement the intent of the `local_*()` calls manually. If you're doing quite a bit of work via `local_*()` or `on.exit()`, this presents a problem. Basically, it's harder to develop with functions that work one way inside a function, but another way in the global environment (or, worse, don't work at all). `substitute()` is another example of this.

As of withr v2.2.0, you can `defer()` events on the global environment, which means that `local_*()` functions work too. This is still a pretty weird thing to do, which is why you get a message about how to trigger execution.


```r
library(withr)

packageVersion("withr")
#> [1] '2.2.0'

defer(print("hi"))
#> Setting deferred event(s) on global environment.
#>   * Execute (and clear) with `deferred_run()`.
#>   * Clear (without executing) with `deferred_clear()`.

deferred_run()
#> [1] "hi"
```

Since the global environment isn't perishable, like a test environment is, you have to call `deferred_run()` explicitly to execute the deferred events. You can also clear them, without running, with `deferred_clear()`.

This new capability is especially handy with self-cleaning test fixtures, like `create_local_package()` and `local_ss()` shown above. Sometimes you have to change global state while developing tests, e.g. change working directory or create test Sheets. But now there's a way to run the associated cleanup on demand.

## Recap

We've demonstrated that it's a problem to change state in a function or test. Obviously there are exceptions if, for example, that is the whole point of the function.

The most common and recommended solution to this is to use `on.exit()` to organize the necessary cleanup, i.e. restore the original state. However, `on.exit()` has some inherent limitations.

If this sort of setup/teardown happens frequently in the functions and tests in a package, it makes sense to write a custom helper. This function should follow the conventions of the `local_*()` functions in withr and will presumably be built around `withr::defer()`.

There is some cost to using a custom `local_*()` helper, as it is one more thing to maintain and that all contributors must understand. Consider whether the pros outweigh the cons when adding another layer of abstraction.
