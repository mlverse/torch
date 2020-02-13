cmake_minimum_required(VERSION 3.5)

project(lanternheders)

############################################################
# YAML
############################################################

include(ExternalProject)
ExternalProject_Add(yamlcpp
    URL https://codeload.github.com/jbeder/yaml-cpp/zip/yaml-cpp-0.6.3
    INSTALL_COMMAND make
)

############################################################
# Generator
############################################################

add_executable(lanterngen
    src/main.cpp
)

target_link_libraries(lanterngen ${CMAKE_DL_LIBS})
