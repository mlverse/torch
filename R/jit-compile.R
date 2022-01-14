#' Compile TorchScript code into a graph
#'
#' See the [TorchScript language reference](https://pytorch.org/docs/stable/jit_language_reference.html#language-reference) for
#' documentation on how to write TorchScript code.
#'
#' @param source valid TorchScript source code.
#'
#' @examples
#' comp <- jit_compile("
#' def fn (x):
#'   return torch.abs(x)
#'
#' def foo (x):
#'   return torch.sum(x)
#'
#' ")
#'
#' comp$fn(torch_tensor(-1))
#' comp$foo(torch_randn(10))
#' @export
jit_compile <- function(source) {
  cpp_jit_compile(source)
}
