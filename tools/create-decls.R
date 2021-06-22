
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
int _lantern_IValue_type (void* self)
bool _lantern_IValue_Bool (void * self)
void* _lantern_IValue_BoolList (void* self)
void* _lantern_IValue_Device (void* self)
double _lantern_IValue_Double (void* self)
void* _lantern_IValue_DoubleList (void* self)
void* _lantern_IValue_Generator (void* self)
void* _lantern_IValue_GenericDict (void* self)
int64_t _lantern_IValue_Int (void* self)
void* _lantern_IValue_IntList (void* self)
void* _lantern_IValue_List (void* self)
void* _lantern_IValue_Module (void* self)
void* _lantern_IValue_Scalar (void* self)
void* _lantern_IValue_String (void* self)
void* _lantern_IValue_Tensor (void* self)
void* _lantern_IValue_TensorList (void* self)
void* _lantern_IValue_Tuple (void* self)
void _lantern_GenericDict_delete (void* x)
void _lantern_GenericList_delete (void* x)
void* _lantern_Stack_at (void* self, int64_t index)
void _lantern_IValue_delete (void* x)
void* _lantern_IValue_from_Bool (bool self)
void* _lantern_IValue_from_BoolList (void* self)
void* _lantern_IValue_from_Device (void* self)
void* _lantern_IValue_from_Double (double self)
void* _lantern_IValue_from_DoubleList (void* self)
void* _lantern_IValue_from_Generator (void* self)
void* _lantern_IValue_from_GenericDict (void* self)
void* _lantern_IValue_from_Int (int64_t self)
void* _lantern_IValue_from_IntList (void* self)
void* _lantern_IValue_from_List (void* self)
void* _lantern_IValue_from_Module (void* self)
void* _lantern_IValue_from_Scalar (void* self)
void* _lantern_IValue_from_String (void* self)
void* _lantern_IValue_from_Tensor (void* self)
void* _lantern_IValue_from_TensorList (void* self)
"  
)

make_decls(decls[-1])
make_load_symbols(decls[-1])
