cmake_minimum_required(VERSION 3.5)

project(lantern)

############################################################
# Helpers
############################################################

function (download_file url filename)
    if(NOT EXISTS ${filename})
        file(DOWNLOAD ${url} ${filename}
             TIMEOUT 300
             TLS_VERIFY ON
        )
    endif()
endfunction()

function (retrieve_lib filename)
    download_file("https://download.pytorch.org/libtorch/cpu/${filename}" "./${filename}")

    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

############################################################
# Library
############################################################

if (APPLE)
    retrieve_lib("libtorch-macos-1.3.0.zip")
endif()
if(LINUX)
    retrieve_lib("libtorch-cxx11-abi-shared-with-deps-1.3.0%2Bcpu.zip")
endif()
if(WIN32)
    retrieve_lib("libtorch-win-shared-with-deps-1.3.0.zip")
endif()

add_library(lantern SHARED 
    src/latern.cpp
)
add_library(lantern::library ALIAS lantern)

target_include_directories(lantern
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)

############################################################
# Tests
############################################################

if (NOT WIN32)
    add_executable(lanterntest
        tests/main.cpp
    )

    target_link_libraries(lanterntest
        PRIVATE 
            lantern::library
    )
endif ()