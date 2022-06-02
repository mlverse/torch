// translate error messages
std::thread::id main_thread_id() noexcept;
std::string translate_error_message(std::string msg) {
  if (std::this_thread::get_id() == main_thread_id()) {
    Rcpp::Function f = Rcpp::Environment::namespace_env("torch").find("translate_error_msg");
    return Rcpp::as<std::string>(f(msg));
  }
  // translation will happen at a later point
  return msg;
}