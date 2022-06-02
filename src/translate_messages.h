// translate error messages
std::string translate_error_message(std::string msg) {
  auto env = Rcpp::Environment::namespace_env("torch");
  Rcpp::Function f = env["translate_error_msg"];
  return Rcpp::as<std::string>(f(msg));
}