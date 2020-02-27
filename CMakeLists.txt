cmake_minimum_required(VERSION 3.5)

project(lantern)

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

function(retrieve_lib filename)
    download_file("https://download.pytorch.org/libtorch/cpu/${filename}" "./${filename}")

    if(NOT EXISTS "libtorch")
        message("-- Extracting ${filename}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endfunction()

############################################################
# Torch
############################################################

if(APPLE)
    retrieve_lib("libtorch-macos-1.3.0.zip")
else()
    if(WIN32)
        retrieve_lib("libtorch-win-shared-with-deps-1.3.0.zip")
    else()
        retrieve_lib("libtorch-cxx11-abi-shared-with-deps-1.3.0%2Bcpu.zip")
    endif()
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_BINARY_DIR}/libtorch/share/cmake/Torch")


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

############################################################
# Library
############################################################

add_library(lantern SHARED 
    src/lantern.cpp
    src/TensorOptions.cpp
    src/Dtype.cpp
    src/Tensor.cpp
    src/Device.cpp
    src/utils.cpp
    src/MemoryFormat.cpp
    src/Generator.cpp
)
add_library(lantern::library ALIAS lantern)

target_include_directories(lantern PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(lantern ${TORCH_LIBRARIES})
set_property(TARGET lantern PROPERTY CXX_STANDARD 14)

############################################################
# Tests
############################################################

add_executable(lanterntest
    tests/init.cpp
    tests/main.cpp
)

target_include_directories(lanterntest PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    tests/
)

target_link_libraries(lanterntest ${CMAKE_DL_LIBS})
