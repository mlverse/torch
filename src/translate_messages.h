// translate error messages
std::string translate_error_message(std::string msg) {
  Rcpp::Function f = Rcpp::Environment::namespace_env("torch").find("translate_error_msg");
  return Rcpp::as<std::string>(f(msg));
}