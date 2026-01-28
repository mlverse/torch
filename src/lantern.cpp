#include <torch.h>
#include <thread>
#include "translate_messages.h"

using namespace Rcpp;

// Defined in autograd.cpp - stores the longjump token when catching LongjumpException
extern SEXP g_longjump_token;

void lantern_host_handler()
{
  if (std::this_thread::get_id() != main_thread_id()) {
    return;
  }
  if (lanternLastError() != NULL) {
    std::string last = lanternLastError();
    lanternLastErrorClear();

    // If we have a stored longjump token, resume the longjump
    if (g_longjump_token != R_NilValue) {
      SEXP token = g_longjump_token;
      g_longjump_token = R_NilValue;
      ::R_ReleaseObject(token);
      ::R_ContinueUnwind(token);
      // R_ContinueUnwind does not return
    }

    std::string error_msg = translate_error_message(std::string(last.c_str()));

    throw Rcpp::exception(error_msg.c_str());
  }
}

bool lantern_loaded = false;
void check_lantern_loaded ()
{
  if (!lantern_loaded)
  {
    throw std::runtime_error("Lantern is not loaded. Please use `install_torch()` to install additional dependencies.");
  }
}

// [[Rcpp::export]]
void cpp_lantern_configure(int log) {
  lanternConfigure(log);
}

// [[Rcpp::export]]
std::string cpp_lantern_version() {
  return std::string(lanternVersion());
}

// [[Rcpp::export]]
void cpp_lantern_init(std::string path) {
  std::string error;
  if (!lanternInit(path, &error))
    Rcpp::stop(error);
}

// [[Rcpp::export]]
void cpp_lantern_test() {
  lanternTest();
}

// [[Rcpp::export]]
bool cpp_lantern_has_error() {
  const char* pLast = lanternLastError();
  return pLast != NULL;
}

// [[Rcpp::export]]
std::string cpp_lantern_last_error() {
  const char* pError = lanternLastError();
  if (pError == NULL)
    return std::string("");
  
  return std::string(pError);
}

// [[Rcpp::export]]
void cpp_lantern_error_clear() {
  lanternLastErrorClear();
}
