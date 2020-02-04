cmake_minimum_required(VERSION 3.5)

project(lantern)

############################################################
# Library
############################################################

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