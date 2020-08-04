library(roxygen2)

.S3method("roxy_tag_parse", "roxy_tag_examples", function(x) {
  roxygen2::tag_examples(x)
})

.S3method("roxy_tag_rd", "roxy_tag_examples", function(x, base_path, env) {
  rd_section("examples", x$val)
})

.S3method("format", "rd_section_examples", function (x, ...) {
  value <- paste0(x$value, collapse = "\n")
  roxygen2:::rd_macro("examples",
    c("if (torch_is_installed()) {", value, "}"),
    space = TRUE
  )
})


devtools::document()