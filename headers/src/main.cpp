#include <fstream>
#include <iostream>
#include <istream>
#include <vector>

#include "yaml-cpp/yaml.h"

void replaceFile(std::string path, std::string start, std::string end, std::string replacement)
{
    std::string line;
    std::ifstream input(path);

    std::vector<std::string> content;
    while (std::getline(input, line))
    {
        content.push_back(line);
    }
    input.close();

    std::ofstream output(path);

    for (auto iter = content.begin(); iter != content.end(); iter++)
    {
        output << *iter << std::endl;
    }
    output.close();
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        std::cout << "Usage: lanterngen declarations.yaml lantern.cpp lantern.h" << std::endl;
        return 1;
    }

    char* pathDeclarations = argv[1];
    char* pathSource = argv[2];
    char* pathHeader = argv[3];

    YAML::Node config = YAML::LoadFile(pathDeclarations);

    replaceFile(pathSource, "/* Autogen Body -- Start */", "/* Autogen Body -- End */", "\n/* content */\n");
    replaceFile(pathHeader, "/* Autogen Headers -- Start */", "/* Autogen Headers -- End */", "\n/* content */\n");

    return 0;
}