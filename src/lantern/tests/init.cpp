#include "init.h"

#include <fstream>
#include <iostream>

#include "lantern/lantern.h"

void myInit(char* path) {
  std::string error;
  std::cout << "Lantern path: " << std::string(path) << std::endl;
  if (!lanternInit(path, &error)) {
    std::cout << "Error: " << error << std::endl;
    throw;
  }
}