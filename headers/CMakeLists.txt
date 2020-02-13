cmake_minimum_required(VERSION 3.5)

project(lanternheders)

############################################################
# Helpers
############################################################

function(download_file url filename)
    if(NOT EXISTS ${filename})
        message("-- Downloading ${filename}")
        file(DOWNLOAD ${url} ${filename}
             TIMEOUT 300
             TLS_VERIFY ON
        )
    endif()
endfunction()

function(install_yaml version)
    download_file("https://codeload.github.com/jbeder/yaml-cpp/zip/yaml-cpp-${version}" "./yaml-cpp-${version}.zip")

    if(NOT EXISTS "yaml-cpp-yaml-cpp-${version}")
        message("-- Extracting YAML")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf "yaml-cpp-${version}.zip"
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endfunction()

############################################################
# YAML
############################################################

install_yaml("0.6.3")

############################################################
# Generator
############################################################

add_executable(lanterngen
    src/main.cpp
)

target_link_libraries(lanterngen ${CMAKE_DL_LIBS})
