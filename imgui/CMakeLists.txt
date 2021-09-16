cmake_minimum_required(VERSION 3.16)
project(imgui)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
SET(IMGUI_DIR="${CMAKE_HOME_DIRECTORY}/imgui")

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${OPENGL_INCLUDE_DIR})


file(GLOB SOURCES *.cpp *.h)
add_library(imgui ${SOURCES})
target_link_libraries(imgui GL GLU glut )

# copy font file to the build directory
# add_custom_command(
#         TARGET imgui POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy
#                 ${CMAKE_SOURCE_DIR}/verdana.ttf
#                 ${CMAKE_CURRENT_BINARY_DIR}/verdana.ttf
#                 )