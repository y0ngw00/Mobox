cmake_minimum_required(VERSION 3.10.2)
project(comcon)

SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_compile_options(-fPIC)
add_compile_options(-std=gnu++14)
add_compile_options(-Wdeprecated-declarations)


find_package(DART REQUIRED COMPONENTS collision-bullet CONFIG)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${DART_INCLUDE_DIRS})
# include_directories(${PYTHON_INCLUDE_DIR})
file(GLOB srcs "*.h" "*.cpp")

add_library(comcon ${srcs})
target_link_libraries(comcon ${DART_LIBRARIES})
