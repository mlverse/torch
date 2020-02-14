#include <iostream>

#include "yaml-cpp/yaml.h"

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: lanterngen declarations.yaml" << std::endl;
        return 1;
    }
    
    YAML::Node config = YAML::LoadFile(argv[1]);

    return 0;
}