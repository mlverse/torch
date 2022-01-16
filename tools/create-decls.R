library(magrittr)

make_decls <- function(decls) {
  type <- stringr::str_match(decls, "(.*) _lantern")[,2]
  name <- stringr::str_match(decls, "(_lantern_[^ ]*) ")[,2]
  args <- stringr::str_match(decls, "(\\(.*\\))")[,2]
  arg_names <- stringr::str_match_all(args, " ([^ ]*)[,\\)]{1}") %>%
    lapply(\(x) x[,2]) %>% sapply(\(x) paste(x, collapse = ", "))

  template <- "
LANTERN_API << type >> (LANTERN_PTR << name >>) << args >>;
HOST_API << type >> << stringr::str_sub(name, 2) >> << args >>
{
  LANTERN_CHECK_LOADED
  << ifelse(type!='void', paste(type, 'ret', '='),'') >> << name >>(<< arg_names >>);
  LANTERN_HOST_HANDLER;
  << ifelse(type!='void', 'return ret;', '') >>
}

"

  glue::glue(template, .open = "<<", .close = ">>")

}

make_load_symbols <- function(decls) {
  name <- stringr::str_match(decls, "(_lantern_[^ ]*) ")[,2]
  glue::glue("LOAD_SYMBOL({name});")
}

decls <- readr::read_lines(
  "
void _lantern_autograd_edge_list_delete (void* x)
void _lantern_autograd_edge_delete (void* x)
"
)

make_decls(decls[-1])
make_load_symbols(decls[-1])
