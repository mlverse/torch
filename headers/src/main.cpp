#include <algorithm>
#include <fstream>
#include <iostream>
#include <istream>
#include <vector>

#include "yaml-cpp/yaml.h"

std::string toLower(std::string str)
{
    std::transform(str.begin(),
                   str.end(),
                   str.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    return str;
}

std::string toCharOnly(std::string str)
{
    auto iterEnd = std::remove_if(str.begin(),
                                  str.end(),
                                  [](unsigned char c){ return !std::isalpha(c); });

    return std::string(str.begin(), iterEnd);
}

std::string toFunction(std::string str, YAML::Node node)
{
    std::string name = "lantern_" + toLower(str);

    for (size_t idx = 0; idx < node.size(); idx++)
    {
        name += "_" + toLower(toCharOnly(node[idx]["dynamic_type"].as<std::string>()))
        ;
    }

    return name;
}

std::string buildArguments(std::string name, YAML::Node node)
{
    std::string arguments = "";

    for (size_t idx = 0; idx < node.size(); idx++)
    {
        if (idx > 0)
        {
            arguments += ", ";
        }

        arguments += "void* " + node[idx]["name"].as<std::string>();
    }

    return arguments;
}

std::string buildCalls(std::string name, YAML::Node node)
{
    std::string arguments = "";

    for (size_t idx = 0; idx < node.size(); idx++)
    {
        if (idx > 0)
        {
            arguments += ", ";
        }

        arguments += "((LanternObject<" + node[idx]["type"].as<std::string>() + ">*)" +
            node[idx]["name"].as<std::string>() + ")->get()";
    }

    return arguments;
}

void replaceFile(std::string path,
                 std::string start,
                 std::string end,
                 std::vector<std::string> replacement)
{
    // read input file
    std::string line;
    std::ifstream input(path);
    std::vector<std::string> content;
    while (std::getline(input, line))
    {
        content.push_back(line);
    }
    input.close();

    // make replacements
    auto iterStart = std::find(content.begin(), content.end(), start);
    if (iterStart != content.end())
    {
        auto iterEnd = std::find(iterStart, content.end(), end);
        if (iterStart != content.end())
        {
            std::cout << "Replacing " << path << std::endl;

            content.erase(iterStart + 1, iterEnd);
            content.insert(iterStart + 1, replacement.begin(), replacement.end());
        }
    }

    // write output file
    std::ofstream output(path);
    for (auto iter = content.begin(); iter != content.end(); iter++)
    {
        output << *iter << std::endl;
    }
    output.close();
}

bool isSupported(YAML::Node node)
{
    if (node["method_of"])
    {
        std::cout << "Skipping " << node["name"].as<std::string>() << std::endl;
        return false;
    }

    return true;
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

    std::cout << "Loaded " << pathDeclarations << " with " << config.size() << " nodes" << std::endl;

    // generate function headers and bodies
    std::vector<std::string> headers;
    std::vector<std::string> bodies;
    for (size_t idx = 0; idx < config.size(); idx++)
    {
        if (!isSupported(config[idx])) continue;

        std::string name = config[idx]["name"].as<std::string>();
        std::string arguments = buildArguments(name, config[idx]["arguments"]);
        std::string function = toFunction(name, config[idx]["arguments"]);
        std::string calls = buildCalls(name, config[idx]["arguments"]);
        std::string returns = config[idx]["returns"][0]["dynamic_type"].as<std::string>();

        headers.push_back("LANTERN_API void* (LANTERN_PTR " + function + ")(" + arguments + ");");
    
        bodies.push_back("void* " + function + "(" + arguments + ")");
        bodies.push_back("{");
        bodies.push_back("    using namespace torch;");
        bodies.push_back("    using namespace at::native;");
        if (returns == "void")
        {
            bodies.push_back("    " + name + "(" + calls + ");");
            bodies.push_back("    return NULL;");
        }
        else
        {
            bodies.push_back("    return (void *) new LanternObject<" + returns + ">(" + name + "(");
            bodies.push_back("        " + calls + "));");
        }
        bodies.push_back("}");
        bodies.push_back("");
    }

    // generate symbol loaders
    std::vector<std::string> symbols;
    for (size_t idx = 0; idx < config.size(); idx ++)
    {
        if (!isSupported(config[idx])) continue;

        std::string name = config[idx]["name"].as<std::string>();
        symbols.push_back("  LOAD_SYMBOL(" + toFunction(name, config[idx]["arguments"]) + ")");
    }

    replaceFile(pathSource, "/* Autogen Body -- Start */", "/* Autogen Body -- End */", bodies);
    replaceFile(pathHeader, "/* Autogen Headers -- Start */", "/* Autogen Headers -- End */", headers);
    replaceFile(pathHeader, "  /* Autogen Symbols -- Start */", "  /* Autogen Symbols -- End */", symbols);

    return 0;
}