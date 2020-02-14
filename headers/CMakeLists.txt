cmake_minimum_required(VERSION 3.5)

project(lanternheders)

############################################################
# YAML
############################################################

include(ExternalProject)
ExternalProject_Add(yamlcpp
    URL https://codeload.github.com/jbeder/yaml-cpp/zip/yaml-cpp-0.6.3
    BUILD_COMMAND make
)

############################################################
# Generator
############################################################

add_executable(lanterngen
    src/main.cpp
)

set_property(TARGET lanterngen PROPERTY CXX_STANDARD 14)

add_dependencies(lanterngen yamlcpp)
target_include_directories(lanterngen PUBLIC
    ${PROJECT_BINARY_DIR}/yamlcpp-prefix/src/yamlcpp/include
)
target_link_libraries(lanterngen ${PROJECT_BINARY_DIR}/yamlcpp-prefix/src/yamlcpp-build/libyaml-cpp.a)
