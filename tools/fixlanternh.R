# Script to fix lantern headers

lantern_h <- here::here("lantern/include/lantern/lantern.h")
lantern <- readLines(lantern_h)

fixed <- NULL
for (entry in lantern) {
  if (grepl("LANTERN_PTR lantern_", entry)) {
    func_name <- stringi::stri_extract(entry, regex = "LANTERN_PTR lantern_[^)]+")
    func_name <- gsub("LANTERN_PTR ", "", func_name)

    params <- stringi::stri_extract(entry, regex = "\\([^\\)]+\\);$")
    if (length(params) == 0 || is.na(params)) params <- ""
    params <- gsub("\\(|\\);", "", params)

    calls <- strsplit(params, ",")[[1]]
    if (length(calls) == 0) calls <- ""
    calls <- stringi::stri_extract(calls, regex = "[a-zA-Z0-9_]+$")
    if (length(calls) == 0 || is.na(calls)) calls <- ""
    calls <- paste(calls, collapse = ", ")

    ret <- stringi::stri_extract(entry, regex = "LANTERN_API [^\\(]+\\(")
    ret <- gsub("LANTERN_API | *\\(", "", ret)

    entry <- c(
      gsub("LANTERN_PTR lantern_", "LANTERN_PTR _lantern_", entry),
      paste0("  HOST_API ", ret, " ", func_name, "(", params, ") { return LANTERN_HOST_HANDLER(_", func_name, "(", calls, ")); }")
    )
  }

  if (length(entry) == 1 && grepl("LOAD_SYMBOL\\(", entry)) {
    func_name <- gsub(".*LOAD_SYMBOL\\(|\\).*", "", entry)
    entry <- gsub("LOAD_SYMBOL\\(.*\\)", paste0("LOAD_SYMBOL(_", func_name, ")"), entry)
  }

  fixed <- c(fixed, entry)
}

writeLines(fixed, lantern_h)
