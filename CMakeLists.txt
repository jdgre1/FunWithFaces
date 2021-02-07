
### ~~~~~~~~~~~~ General  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cmake_minimum_required(VERSION 3.10)

project( FaceApp )

set(CMAKE_CXX_STANDARD 17)


### ~~~~~~~~~~~~ Find Packages ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

find_package(OpenCV 4.0 REQUIRED)


### ~~~~~~~~~~~~ General Include/Add ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# IMPORTANT - "include_directories" must come before "add_subdirectory"!
include_directories(include ${OpenCV_INCLUDE_DIR})

add_subdirectory(src)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
