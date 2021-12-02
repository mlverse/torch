#include <algorithm>
#include <fstream>
#include <iostream>
#include <istream>
#include <vector>
#include <regex>

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

std::string removeAt(std::string str)
{
    std::regex e ("at::"); 
    auto result = std::regex_replace (str, e, "");
    return std::regex_replace(result, std::regex("const Scalar &"), "Scalar");
}

std::string toFunction(std::string str, YAML::Node node)
{
    std::string name = toLower(str);

    for (size_t idx = 0; idx < node.size(); idx++)
    {
        name += "_" + toLower(toCharOnly(removeAt(node[idx]["dynamic_type"].as<std::string>())));
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

std::string buildArgumentsCalls(std::string name, YAML::Node node)
{
    std::string arguments = "";

    for (size_t idx = 0; idx < node.size(); idx++)
    {
        if (idx > 0)
        {
            arguments += ", ";
        }

        arguments += node[idx]["name"].as<std::string>();
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
    objects.push_back("ArrayRef");

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
    if (type == "Device" | type == "std::vector<torch::Dimname>")
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

        std::string type = removeAt(node[idx]["dynamic_type"].as<std::string>());
        std::string dtype = removeAt(node[idx]["type"].as<std::string>());

        if (type == "ArrayRef<double>" & dtype != "c10::optional<ArrayRef<double>>")
        {
          type = "std::vector<double>";
        }
        else if (type == "const c10::List<c10::optional<Tensor>> &")
        {
            type = "c10::List<c10::optional<torch::Tensor>>";
        }
        else if (type == "Generator *")
        {
            type = "std::shared_ptr<torch::Generator>";
        }
        else if (type == "Stream")
        {
            type = "at::Stream";
        }
        else if (dtype == "const c10::optional<Tensor> &")
        {
            type = "c10::optional<torch::Tensor>";
        }
        else if (type == "const at::Scalar &" && dtype == "const c10::optional<at::Scalar> &")
        {
            type = "c10::optional<at::Scalar>";
        }

        // add optional call if required
        std::string call = node[idx]["name"].as<std::string>();
        if ((dtype.find("c10::optional") != std::string::npos) & 
            (type != "std::vector<torch::Dimname>") &
            (type != "c10::List<c10::optional<torch::Tensor>>") &
            (type != "c10::optional<torch::Tensor>") &
            (type != "const c10::List<c10::optional<at::Tensor>> &") &
            (type != "c10::optional<at::Scalar>"))
        {
            if ((type != "double") & 
                (type != "IntArrayRef") &
                (type != "ArrayRef<double>") &
                type  != "int64_t" &
                type  != "DimnameList" &
                type  != "Generator"
                )
            {
                call = "optional<" + addNamespace(type) + ">(" + call + ").get()";
            }
            
            type = "c10::optional<" + type + ">";
        }


        
        if (type == "Tensor")
        {
            arguments += "from_raw::Tensor(" + call + ")";
        }
        else if (type == "TensorList")
        {
            arguments += "from_raw::TensorList(" + call + ")";
        }
        else if (type == "ScalarType")
        {
            arguments += "from_raw::ScalarType(" + call + ")";
        }
        else if (type == "Scalar")
        {
            arguments += "from_raw::Scalar(" + call + ")";
        }
        else if (type == "TensorOptions")
        {
            arguments += "from_raw::TensorOptions(" + call + ")";
        }
        else if (type == "Device")
        {
            arguments += "from_raw::Device(" + call + ")";
        }
        else if (type == "Dimname")
        {
            arguments += "from_raw::Dimname(" + call + ")";
        }
        else if (type == "DimnameList")
        {
            arguments += "from_raw::DimnameList(" + call + ")";
        }
        else if (type == "c10::optional<DimnameList>")
        {
            arguments += "from_raw::optional::DimnameList(" + call + ")";
        }
        else if (type == "Generator")
        {
            arguments += "from_raw::Generator(" + call + ")";
        }
        else if (type == "c10::optional<Generator>")
        {
            arguments += "from_raw::optional::Generator(" + call + ")";
        }
        else if (type == "IntArrayRef")
        {
            arguments += "from_raw::IntArrayRef(" + call + ")";
        }
        else if (type == "Storage")
        {
            arguments += "from_raw::Storage(" + call + ")";
        }
        else if (type == "std::string")
        {
            arguments += "from_raw::string(" + call + ")";
        }
        else if (type == "int64_t")
        {
            arguments += "from_raw::int64_t(" + call + ")";
        }
        else
        {
            arguments += "((" + lanternObject(type) + "<" + addNamespace(type) + ">*)" +
                     call + ")->get()";
        }

        
        if (type == "std::shared_ptr<torch::Generator>")
        {
            arguments += ".get()";
        }
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

        type += addNamespace(removeAt(node[idx]["dynamic_type"].as<std::string>()));
    }

    if (node.size() > 1)
    {
        type = "std::vector<void*>";
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
        std::string argumentsCalls = buildArgumentsCalls(name, config[idx]["arguments"]);
        std::string function = toFunction(name, config[idx]["arguments"]);
        std::string returns = buildReturn(config[idx]["returns"]);

        std::string calls = "";
        std::string functionCall = "";
        if (hasMethodOf(config[idx], "namespace"))
        {
            headers.push_back("  LANTERN_API void* (LANTERN_PTR _lantern_" + function + ")(" + arguments + ");");
            headers.push_back("  HOST_API void* lantern_" + function + "(" + arguments + ") { LANTERN_CHECK_LOADED void* ret = _lantern_" + function + "(" + argumentsCalls + "); LANTERN_HOST_HANDLER return ret; }");
  
            calls = buildCalls(name, config[idx]["arguments"], 0);
            functionCall = "torch::";

            bodies.push_back("void* _lantern_" + function + "(" + arguments + ")");
            bodies.push_back("{");
            bodies.push_back("  LANTERN_FUNCTION_START");
            if (returns == "void" | (config[idx]["returns"].size() == 0))
            {
                bodies.push_back("    " + functionCall + name + "(" + calls + ");");
                bodies.push_back("    return NULL;");
            }
            else
            {
                if (config[idx]["returns"].size() == 1)
                {
                    if (returns == "torch::Tensor")
                    {
                        bodies.push_back("    return make_unique::Tensor(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::TensorList")
                    {
                        bodies.push_back("    return make_unique::TensorList(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::ScalarType")
                    {
                        bodies.push_back("    return make_unique::ScalarType(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::Scalar")
                    {
                        bodies.push_back("    return make_unique::Scalar(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::TensorOptions")
                    {
                        bodies.push_back("    return make_unique::TensorOptions(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::DimnameList")
                    {
                        bodies.push_back("    return make_unique::DimnameList(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::IntArrayRef")
                    {
                        bodies.push_back("    return make_unique::IntArrayRef(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::QScheme")
                    {
                        bodies.push_back("    return make_unique::QScheme(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::Storage")
                    {
                        bodies.push_back("    return make_unique::Storage(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "int64_t")
                    {
                        bodies.push_back("    return make_unique::int64_t(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else
                    {
                        bodies.push_back("    return (void *) new LanternObject<" + returns + ">(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                }
                else
                {
                    bodies.push_back("    return (void *) new LanternObject<" + returns + ">(to_vector(" + functionCall + name + "(");
                    bodies.push_back("        " + calls + ")));");
                }
            }
            bodies.push_back("  LANTERN_FUNCTION_END");
            bodies.push_back("}");
            bodies.push_back("");
        }

        calls = "";
        functionCall = "";
        if (hasMethodOf(config[idx], "Tensor") || name == "stride")
        {

            headers.push_back("  LANTERN_API void* (LANTERN_PTR _lantern_Tensor_" + function + ")(" + arguments + ");");
            headers.push_back("  HOST_API void* lantern_Tensor_" + function + "(" + arguments + ") { void* ret = _lantern_Tensor_" + function + "(" + argumentsCalls + "); LANTERN_HOST_HANDLER return ret; }");
  
            calls = buildCalls(name, config[idx]["arguments"], 1);

            std::string firstType = config[idx]["arguments"][0]["dynamic_type"].as<std::string>();
            std::string firstName = config[idx]["arguments"][0]["name"].as<std::string>();

            if (firstType == "at::Tensor" || firstType == "Tensor" || firstType == "torch::Tensor")
            {
                functionCall = "from_raw::Tensor(" + firstName + ").";
            }
            else
            {
                functionCall = "((LanternObject<" + addNamespace(firstType) + ">*)" + firstName + ")->get().";
            }

            bool skipCuda102 = function == "true_divide_tensor_scalar" ||
                               function == "true_divide__tensor_scalar" ||
                               function == "true_divide_tensor_tensor" ||
                               function == "true_divide__tensor_tensor";


            bodies.push_back("void* _lantern_Tensor_" + function + "(" + arguments + ")");
            bodies.push_back("{");
            bodies.push_back("  LANTERN_FUNCTION_START");
            if (skipCuda102)
            {
                bodies.push_back("#ifdef CUDA102");
                bodies.push_back("    throw \"Not Implemented\";");
                bodies.push_back("#else");
            }
            if (returns == "void" | (config[idx]["returns"].size() == 0))
            {
                bodies.push_back("    " + functionCall + name + "(" + calls + ");");
                bodies.push_back("    return NULL;");
            }
            else
            {
                if (config[idx]["returns"].size() == 1)
                {
                    if (returns == "torch::Tensor")
                    {
                        bodies.push_back("    return make_unique::Tensor(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::TensorList")
                    {
                        bodies.push_back("    return make_unique::TensorList(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::ScalarType")
                    {
                        bodies.push_back("    return make_unique::ScalarType(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::Scalar")
                    {
                        bodies.push_back("    return make_unique::Scalar(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::TensorOptions")
                    {
                        bodies.push_back("    return make_unique::TensorOptions(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::DimnameList")
                    {
                        bodies.push_back("    return make_unique::DimnameList(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::IntArrayRef")
                    {
                        bodies.push_back("    return make_unique::IntArrayRef(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::QScheme")
                    {
                        bodies.push_back("    return make_unique::QScheme(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "torch::Storage")
                    {
                        bodies.push_back("    return make_unique::Storage(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else if (returns == "int64_t")
                    {
                        bodies.push_back("    return make_unique::int64_t(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                    else
                    {
                        bodies.push_back("    return (void *) new LanternObject<" + returns + ">(" + functionCall + name + "(");
                        bodies.push_back("        " + calls + "));");
                    }
                }
                else
                {
                    bodies.push_back("    return (void *) new LanternObject<" + returns + ">(to_vector(" + functionCall + name + "(");
                    bodies.push_back("        " + calls + ")));");
                }
            }
            if (skipCuda102)
            {
                bodies.push_back("#endif");
            }
            bodies.push_back("  LANTERN_FUNCTION_END");
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
            symbols.push_back("  LOAD_SYMBOL(_lantern_" + toFunction(name, config[idx]["arguments"]) + ")");
        }

        if (hasMethodOf(config[idx], "Tensor") || name == "stride")
        {
            symbols.push_back("  LOAD_SYMBOL(_lantern_Tensor_" + toFunction(name, config[idx]["arguments"]) + ")");
        }
    }

    replaceFile(pathSource, "/* Autogen Body -- Start */", "/* Autogen Body -- End */", bodies);
    replaceFile(pathHeader, "  /* Autogen Headers -- Start */", "  /* Autogen Headers -- End */", headers);
    replaceFile(pathHeader, "  /* Autogen Symbols -- Start */", "  /* Autogen Symbols -- End */", symbols);

    return 0;
}