cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 OLD)
cmake_policy(SET CMP0002 OLD)


## Set VAR to ARGN unless it already has some value
macro(default VAR) # ARGN
  if(NOT DEFINED ${VAR})
    set(${VAR} ${ARGN})
  endif()
endmacro()


## Set VAR to a list with elements prefixed by PREFIX
#
# Useful for prefixing a common path onto multiple glob arguments.
macro(prefix VAR PREFIX) # ARGN
  string(REGEX REPLACE "([^;]+)" "${PREFIX}\\1" ${VAR} "${ARGN}")
endmacro()

## Append trailing strings to VAR without delimiters
#
# This function is similar to @c list(APPEND) but concatenates input
# arguments without semicolon delimiters.  Unlike @c set(VAR), this
# function permits multi-line concatenations.
macro(append VAR) # ARGN
  string(REPLACE "" "" ${VAR} "${${VAR}}" ${ARGN})
endmacro()


## Set VAR to the list of all source files beneath DIR
macro(glob_sources VAR DIR)
  prefix(${VAR} ${DIR}/ ${SRC_GLOBS})
  file(GLOB_RECURSE ${VAR} FOLLOW_SYMLINKS ${${VAR}})
endmacro()

## Find and include all scripts matching listed globs
macro(include_glob) # ARGN
  file(GLOB gcm__FILES ${ARGN})
  foreach(gcm__FILE ${gcm__FILES})
    include(${gcm__FILE})
  endforeach()
endmacro()

macro(GenCMake_configure TARGET_PATH NAME_VAR SOURCES_VAR)
  if(IS_DIRECTORY ${TARGET_PATH})
    # Directory target; glob for scripts/sources
    include_glob(${TARGET_PATH}/${SCRIPT_GLOB})
    glob_sources(${SOURCES_VAR} ${TARGET_PATH})
  else()
    # Source file target; no scripts to be found
    set(${SOURCES_VAR} ${TARGET_PATH})
  endif()
  # Strip path parents and possible extension for target name
  get_filename_component(${NAME_VAR} ${TARGET_PATH} NAME_WE)

  # Ignore empty targets
  if(NOT ${SOURCES_VAR})
    message(WARNING "Target path ${TARGET_PATH} contains no sources")
    return()
  endif()
endmacro()

project(PAPERBOAT CXX C Fortran)
# File globs and regex patterns
default(SCRIPT_GLOB *.gen.cmake CACHE STRING
  "Used to find/include scripts for linking libraries, etc.")
default(SCRIPT_PAT "[.]gen[.]cmake(/|$)" CACHE STRING
  "Used to find/include scripts for linking libraries, etc.")
default(SRC_GLOBS *.c *.C *.c++ *.cc *.cpp *.cxx CACHE STRING
  "Globs matching source files built by this project.")

include_directories(${CMAKE_SOURCE_DIR}/../../karnagio/include)
include_directories(${CMAKE_SOURCE_DIR}/../../karnagio/src/lib)
include_directories(${CMAKE_SOURCE_DIR}/../)

GenCMake_configure(${CMAKE_SOURCE_DIR} paperboat SOURCES)
add_library(paperboat ${SOURCES})
set(GenCMake_LIBRARIES "")
set(GenCMake_VERSION "0.6.3")
include(Boost.gen.cmake)
include(BoostThread.gen.cmake)
include(LAPACK.gen.cmake)
include(PTHREADS.gen.cmake)
include(Trilinos.gen.cmake)
target_link_libraries(paperboat ${GenCMake_LIBRARIES})

append(CMAKE_CXX_FLAGS " -Wall -Wno-sign-compare -fno-strict-aliasing -fPIC")
