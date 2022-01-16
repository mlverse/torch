#include <regex>

std::string translate_dim_error_msg(std::string msg) {
  auto regex = std::regex(
      "(?:.|\\r?\\n)*Dimension out of range \\(expected to be in range of "
      "\\[-[0-9]+, ([0-9]+)\\], but got (-?[0-9]+)\\)(?:.|\\r?\\n)*");
  std::smatch m;

  if (std::regex_match(msg, m, regex)) {
    auto l = msg.length();
    msg.replace(m.position(1), m.length(1),
                std::to_string(std::stoi(m[1].str()) + 1));

    // treat the case when we get 9 and we inscreased the lenght of
    // the string by 1.
    int d = 0;
    if (l < msg.length()) {
      d = msg.length() - l;
    }

    auto i = std::stoi(msg.substr(m.position(2) + d, m.length(2)));

    if (i > 0) {
      i = i + 1;
    }

    msg.replace(m.position(2) + d, m.length(2), std::to_string(i));
  }

  return msg;
}

std::string translate_dim_size_error_msg(std::string msg) {
  auto regex = std::regex(
      "(?:.|\\r?\\n)*dimension ([0-9]+) does not have size "
      "[0-9]+(?:.|\\r?\\n)*");
  std::smatch m;

  if (std::regex_match(msg, m, regex)) {
    msg.replace(m.position(1), m.length(1),
                std::to_string(std::stoi(m[1].str()) + 1));
  }

  return msg;
}

std::string translate_max_index_msg(std::string msg) {
  auto regex = std::regex(
      "(?:.|\\r?\\n)*Found an invalid max index: ([0-9]+)(?:.|\\r?\\n)*");
  std::smatch m;

  if (std::regex_match(msg, m, regex)) {
    msg.replace(m.position(1), m.length(1),
                std::to_string(std::stoi(m[1].str()) + 1));
  }

  return msg;
}

std::string translate_index_out_of_range_msg(std::string msg) {
  auto regex = std::regex(
      "(?:.|\\r?\\n)*index ([0-9]+) out of range for tensor of size "
      "\\[[0-9]+\\] at dimension ([0-9]+)(?:.|\\r?\\n)*");
  std::smatch m;

  if (std::regex_match(msg, m, regex)) {
    auto l = msg.length();
    msg.replace(m.position(1), m.length(1),
                std::to_string(std::stoi(m[1].str()) + 1));

    // treat the case when we get 9 and we inscreased the lenght of
    // the string by 1.
    int d = 0;
    if (l < msg.length()) {
      d = msg.length() - l;
    }

    auto i = std::stoi(msg.substr(m.position(2) + d, m.length(2)));

    if (i >= 0) {
      i = i + 1;
    }

    msg.replace(m.position(2) + d, m.length(2), std::to_string(i));
  }

  return msg;
}

std::string translate_target_index_msg(std::string msg) {
  auto regex = std::regex(
      "(?:.|\\r?\\n)*Target (-?[0-9]+) is out of bounds.(?:.|\\r?\\n)*");
  std::smatch m;

  if (std::regex_match(msg, m, regex)) {
    msg.replace(m.position(1), m.length(1),
                std::to_string(std::stoi(m[1].str()) + 1));
  }

  return msg;
}

std::string translate_contract_error_msg(std::string msg) {
  auto regex = std::regex(
      "(?:.|\\r?\\n)*contracted dimensions need to match, but first has size "
      "[0-9]+ in dim ([0-9]+) and second has size [0-9]+ in dim "
      "([0-9]+)(?:.|\\r?\\n)*");
  std::smatch m;

  if (std::regex_match(msg, m, regex)) {
    auto l = msg.length();
    msg.replace(m.position(1), m.length(1),
                std::to_string(std::stoi(m[1].str()) + 1));

    // treat the case when we get 9 and we increased the lenght of
    // the string by 1.
    int d = 0;
    if (l < msg.length()) {
      d = msg.length() - l;
    }

    auto i = std::stoi(msg.substr(m.position(2) + d, m.length(2)));

    if (i >= 0) {
      i = i + 1;
    }

    msg.replace(m.position(2) + d, m.length(2), std::to_string(i));
  }

  return msg;
}

// translate error messages
std::string translate_error_message(std::string msg) {
  std::string out;
  out = translate_dim_size_error_msg(msg);
  out = translate_dim_error_msg(out);
  out = translate_max_index_msg(out);
  out = translate_index_out_of_range_msg(out);
  out = translate_target_index_msg(out);
  out = translate_contract_error_msg(out);
  return out;
}