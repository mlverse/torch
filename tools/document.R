library(roxygen2)


# Patch examples ----------------------------------------------------------
# we path examples so we add the `if` conditional.

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

# Subsection --------------------------------------------------------------

.S3method("roxy_tag_parse", "roxy_tag_subsection", function(x) {
  roxygen2::tag_markdown(x)
})

.S3method("roxy_tag_rd", "roxy_tag_subsection", function(x, base_path, env) {
  pieces <- stringr::str_split(x$val, ":", n = 2)[[1]]
  title <- stringr::str_split(pieces[1], "\n")[[1]]

  if (length(title) > 1) {
    roxygen2:::roxy_tag_warning(x, paste0(
      "Section title spans multiple lines.\n",
      "Did you forget a colon (:) at the end of the title?"
    ))
    return()
  }

  roxygen2:::rd_section_section(pieces[1], pieces[2])
})

.S3method("format", "rd_section_subsection", function(x, ...) {
  paste0(
    "\\subsection{", x$value$title, "}{\n", x$value$content, "\n}\n",
    collapse = "\n"
  )
})

# Params ------------------------------------------------------------------
# avoids using the \arguments{} so we can have two sections with parameters
# for different signatures.




devtools::document()