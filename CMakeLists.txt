cmake_minimum_required(VERSION 3.10.2)
project(com)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++17)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-lm)

add_compile_options(-DHAVE_CSTDDEF)

add_definitions(-DROOT_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")
set(Boost_NO_BOOST_CMAKE ON) 


find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)	
find_package(Boost 1.71 EXACT COMPONENTS filesystem python36 numpy36 program_options)
IF(NOT Boost_FOUND)
	find_package(Boost 1.71 EXACT COMPONENTS filesystem python38 numpy38 program_options)
ENDIF()

add_subdirectory( sim )
add_subdirectory( python )
add_subdirectory( render )
add_subdirectory( imgui )