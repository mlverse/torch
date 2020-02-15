cmake_minimum_required(VERSION 3.5)

project(lanternheders)

############################################################
# YAML
############################################################

include(ExternalProject)
ExternalProject_Add(yamlcpp
    URL https://codeload.github.com/jbeder/yaml-cpp/zip/yaml-cpp-0.6.3
    BUILD_COMMAND make
    STEP_TARGETS build
)
ExternalProject_Get_Property(yamlcpp install_dir)
set(yaml_install_dir ${install_dir})

############################################################
# Generator
############################################################

add_executable(lanterngen
    src/main.cpp
)

set_property(TARGET lanterngen PROPERTY CXX_STANDARD 14)

add_dependencies(lanterngen yamlcpp)
include_directories(${yaml_install_dir}/src/yamlcpp/include)
target_link_libraries(lanterngen ${yaml_install_dir}/src/yamlcpp-build/libyaml-cpp.a)
