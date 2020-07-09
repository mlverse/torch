#include "torch_types.h"
#include <regex>

using namespace Rcpp;

std::string translate_dim_error_msg (std::string msg)
{
  auto regex = std::regex("(?:.|\\r?\\n)*Dimension out of range \\(expected to be in range of \\[-[0-9]+, ([0-9]+)\\], but got ([0-9]+)\\)(?:.|\\r?\\n)*");
  std::smatch m;
  
  if (std::regex_match(msg, m, regex))
  {
    auto l = msg.length();
    msg.replace(m.position(1), m.length(1), std::to_string(std::stoi(m[1].str()) + 1));
    
    // treat the case when we get 9 and we inscreased the lenght of 
    // the string by 1.
    int d = 0;
    if (l < msg.length())
    {
      d = msg.length() - l;
    }
    
    auto i = msg.substr(m.position(2) + d, m.length(2));
    msg.replace(m.position(2) + d, m.length(2), std::to_string(std::stoi(i) + 1));
  }
  
  return msg;
}

// translate error messages
std::string translate_error_message (std::string msg)
{
  return translate_dim_error_msg(msg);
}

void lantern_host_handler()
{
  if (lanternLastError() != NULL) {
    std::string last = lanternLastError();
    lanternLastErrorClear();
    
    std::string error_msg = translate_error_message(std::string(last.c_str()));
    
    throw Rcpp::exception(error_msg.c_str());
  } 
}

// [[Rcpp::export]]
void cpp_lantern_configure(bool log) {
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
