#include <thread>

std::thread::id main_thread_id() noexcept
{
  static const auto tid = std::this_thread::get_id();

  return tid;
}
