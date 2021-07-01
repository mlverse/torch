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
void* _lantern_jit_compile_get_method (void* cu, void* name)
void* _lantern_jit_compile_list_methods (void* cu)
void* _lantern_jit_compile (void* source, void* cu)
"  
)

make_decls(decls[-1])
make_load_symbols(decls[-1])
