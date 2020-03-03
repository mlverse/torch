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
                   [](unsigned char c) { return std::tolower(c); });

    return str;
}

std::string toCharOnly(std::string str)
{
    auto iterEnd = std::remove_if(str.begin(),
                                  str.end(),
                                  [](unsigned char c) { return !std::isalpha(c); });

    return std::string(str.begin(), iterEnd);
}

std::string toFunction(std::string str, YAML::Node node)
{
    std::string name = toLower(str);

    for (size_t idx = 0; idx < node.size(); idx++)
    {
        name += "_" + toLower(toCharOnly(node[idx]["dynamic_type"].as<std::string>()));
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

std::string addNamespace(std::string name)
{
    std::vector<std::string> objects;
    objects.push_back("Tensor");
    objects.push_back("Scalar");
    objects.push_back("Generator");
    objects.push_back("Dimname");
    objects.push_back("IntArrayRef");
    objects.push_back("MemoryFormat");
    objects.push_back("ConstQuantizerPtr");
    objects.push_back("Storage");
    objects.push_back("Device");
    objects.push_back("QScheme");

    for (auto iter = objects.begin(); iter != objects.end(); iter++)
    {
        std::size_t found = name.find(*iter);
        if (found != std::string::npos && (found == 0 || name.at(found - 1) == ' ' || name.at(found - 1) == '<'))
            name = name.insert(found, "torch::");
    }

    return name;
}

std::string lanternObject(std::string type)
{
    if (type == "Device")
    {
        return "LanternPtr";
    }
    else
    {
        return "LanternObject";
    }
}

std::string buildCalls(std::string name, YAML::Node node, size_t start)
{
    std::string arguments = "";

    for (size_t idx = start; idx < node.size(); idx++)
    {
        if (idx > start)
        {
            arguments += ", ";
        }

        std::string type = node[idx]["dynamic_type"].as<std::string>();
        arguments += "((" + lanternObject(type) + "<" + addNamespace(type) + ">*)" +
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

bool hasMethodOf(YAML::Node node, std::string method)
{
    std::string name = node["name"].as<std::string>();

    if (node["method_of"])
    {
        for (size_t i = 0; i < node["method_of"].size(); i++)
            if (node["method_of"][i].as<std::string>() == method)
                return true;
    }

    return false;
}

bool isSupported(YAML::Node node)
{
    std::string name = node["name"].as<std::string>();

    if (!hasMethodOf(node, "namespace") && !hasMethodOf(node, "Tensor"))
    {
        std::cout << "Skipping (methodof) " << name << std::endl;
        return false;
    }

    if (node["name"].as<std::string>() == "normal")
    {
        std::cout << "Skipping (conversion) " << name << std::endl;
        return false;
    }

    if (node["name"].as<std::string>() == "polygamma")
    {
        std::cout << "Skipping (conversion) " << name << std::endl;
        return false;
    }

    return true;
}

std::string buildReturn(YAML::Node node)
{
    std::string type = "";
    for (size_t idx = 0; idx < node.size(); idx++)
    {
        if (idx > 0)
            type += ", ";

        type += addNamespace(node[idx]["dynamic_type"].as<std::string>());
    }

    if (node.size() > 1)
    {
        type = "std::tuple<" + type + ">";
    }

    return type;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "Usage: lanterngen declarations.yaml lantern.cpp lantern.h" << std::endl;
        return 1;
    }

    char *pathDeclarations = argv[1];
    char *pathSource = argv[2];
    char *pathHeader = argv[3];

    YAML::Node config = YAML::LoadFile(pathDeclarations);

    std::cout << "Loaded " << pathDeclarations << " with " << config.size() << " nodes" << std::endl;

    // generate function headers and bodies
    std::vector<std::string> headers;
    std::vector<std::string> bodies;
    for (size_t idx = 0; idx < config.size(); idx++)
    {
        if (!isSupported(config[idx]))
            continue;

        std::string name = config[idx]["name"].as<std::string>();
        std::string arguments = buildArguments(name, config[idx]["arguments"]);
        std::string function = toFunction(name, config[idx]["arguments"]);
        std::string returns = buildReturn(config[idx]["returns"]);

        std::string calls = "";
        std::string functionCall = "";
        if (hasMethodOf(config[idx], "namespace"))
        {
            headers.push_back("LANTERN_API void* (LANTERN_PTR lantern_" + function + ")(" + arguments + ");");

            calls = buildCalls(name, config[idx]["arguments"], 0);
            functionCall = "torch::";

            bodies.push_back("void* lantern_" + function + "(" + arguments + ")");
            bodies.push_back("{");
            if (returns == "void")
            {
                bodies.push_back("    " + functionCall + name + "(" + calls + ");");
                bodies.push_back("    return NULL;");
            }
            else
            {
                bodies.push_back("    return (void *) new LanternObject<" + returns + ">(" + functionCall + name + "(");
                bodies.push_back("        " + calls + "));");
            }
            bodies.push_back("}");
            bodies.push_back("");
        }

        calls = "";
        functionCall = "";
        if (hasMethodOf(config[idx], "Tensor"))
        {
            headers.push_back("LANTERN_API void* (LANTERN_PTR lantern_Tensor_" + function + ")(" + arguments + ");");

            calls = buildCalls(name, config[idx]["arguments"], 1);

            std::string firstType = config[idx]["arguments"][0]["type"].as<std::string>();
            std::string firstName = config[idx]["arguments"][0]["name"].as<std::string>();
            functionCall = "((LanternObject<" + addNamespace(firstType) + ">*)" + firstName + ")->get().";

            bodies.push_back("void* lantern_Tensor_" + function + "(" + arguments + ")");
            bodies.push_back("{");
            if (returns == "void")
            {
                bodies.push_back("    " + functionCall + name + "(" + calls + ");");
                bodies.push_back("    return NULL;");
            }
            else
            {
                bodies.push_back("    return (void *) new LanternObject<" + returns + ">(" + functionCall + name + "(");
                bodies.push_back("        " + calls + "));");
            }
            bodies.push_back("}");
            bodies.push_back("");
        }
    }

    // generate symbol loaders
    std::vector<std::string> symbols;
    for (size_t idx = 0; idx < config.size(); idx++)
    {
        if (!isSupported(config[idx]))
            continue;

        std::string name = config[idx]["name"].as<std::string>();

        if (hasMethodOf(config[idx], "namespace"))
        {
            symbols.push_back("  LOAD_SYMBOL(lantern_" + toFunction(name, config[idx]["arguments"]) + ")");
        }

        if (hasMethodOf(config[idx], "Tensor"))
        {
            symbols.push_back("  LOAD_SYMBOL(lantern_Tensor_" + toFunction(name, config[idx]["arguments"]) + ")");
        }
    }

    replaceFile(pathSource, "/* Autogen Body -- Start */", "/* Autogen Body -- End */", bodies);
    replaceFile(pathHeader, "/* Autogen Headers -- Start */", "/* Autogen Headers -- End */", headers);
    replaceFile(pathHeader, "  /* Autogen Symbols -- Start */", "  /* Autogen Symbols -- End */", symbols);

    return 0;
}