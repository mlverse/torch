test_that("array-or-string dispatch retains a fallback across locales", {
  decls <- declarations()
  groups <- lapply(c("conv1d", "conv2d", "conv3d"), function(name) {
    Filter(function(x) x$name == name && "namespace" %in% x$method_of, decls)
  })
  generate <- function() {
    vapply(groups, function(group) {
      expect_true(torchgen:::is_guarded_array_string_dispatch(group, "namespace"))
      as.character(torchgen:::r_namespace_body(group))
    }, "")
  }

  old_locale <- Sys.getlocale("LC_COLLATE")
  on.exit(Sys.setlocale("LC_COLLATE", old_locale), add = TRUE)
  original <- generate()
  Sys.setlocale("LC_COLLATE", "C")
  expect_identical(generate(), original)
  expect_true(all(grepl("cpp_torch_dispatch_namespace_", original, fixed = TRUE)))
})
