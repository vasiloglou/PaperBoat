## GenCMake: Generic CMake project configuration script
#
# @file   GenCMake.cmake
# @author Ryan N. Riegel
# @date   2010-04-05
#
# This CMake script automatically constructs a project by scanning the
# source directory tree for specially named files and directories.
#
#   - The project name and version are extracted from the source
#     directory's parent directory with format '<name>-<version>'.
#
#   - Scripts matching *.gen.cmake in the source directory root are
#     automatically included before build target generation.  These
#     may call @c include_directories(), modify CMAKE_CXX_FLAGS, etc.,
#     or extend GenCMake_LIBRARIES, which are linked by all targets.
#
#   - Directories and source files in the 'lib' subdirectory define
#     library targets 'lib<basename>'.  Directory targets compose all
#     (recursively) contained sources and further scan for contained
#     *.gen.cmake scripts defining target-specific flags and library
#     dependencies, etc.
#
#   - Directories and source files in 'bin' define executable targets
#     '<basename>' and are otherwise handled similarly to libraries.
#
#   - Directories and source files in 'test' define unit test targets
#     '<basename>-test', which are collectively compiled with special
#     target 'all-tests' rather than default target 'all'.  Tests are
#     registered with CTest and run with arguments GenCMake_TEST_ARGS.
#
# Public header files should not be maintained beneath CMake's source
# directory but instead in sibling directory include, which is copied
# during installation and packaging.  Standard CMake-generated targets
# 'test', 'install', 'package', and 'package_source' are available for
# their respective tasks, however, see the parent directory's Makefile
# for additional support with these operations.

cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 OLD)
cmake_policy(SET CMP0002 OLD)


#
# Utility functions
#

## Print a message if VERBOSE true in calling scope
macro(verbose) # ARGN
  if(VERBOSE)
    message(${ARGN})
  endif()
endmacro()

## Set VAR to ARGN unless it already has some value
macro(default VAR) # ARGN
  if(NOT DEFINED ${VAR})
    set(${VAR} ${ARGN})
  endif()
endmacro()

## Append trailing strings to VAR without delimiters
#
# This function is similar to @c list(APPEND) but concatenates input
# arguments without semicolon delimiters.  Unlike @c set(VAR), this
# function permits multi-line concatenations.
macro(append VAR) # ARGN
  string(REPLACE "" "" ${VAR} "${${VAR}}" ${ARGN})
endmacro()

## Set VAR to a list with elements prefixed by PREFIX
#
# Useful for prefixing a common path onto multiple glob arguments.
macro(prefix VAR PREFIX) # ARGN
  string(REGEX REPLACE "([^;]+)" "${PREFIX}\\1" ${VAR} "${ARGN}")
endmacro()

## Set VAR to the list of all source files beneath DIR
macro(glob_sources VAR DIR)
  prefix(${VAR} ${DIR}/ ${SRC_GLOBS})
  file(GLOB_RECURSE ${VAR} FOLLOW_SYMLINKS ${${VAR}})
endmacro()

## Set VAR to globbed files not matching IGNORED_REGEX
macro(glob_exclude VAR IGNORED_REGEX) # ARGN
  file(GLOB gcm__FILES ${ARGN})
  foreach(gcm__FILE ${gcm__FILES})
    if(NOT gcm__FILE MATCHES ${IGNORED_REGEX})
      list(APPEND ${VAR} ${gcm__FILE})
    endif()
  endforeach()
endmacro()

## Find and include all scripts matching listed globs
macro(include_glob) # ARGN
  file(GLOB gcm__FILES ${ARGN})
  foreach(gcm__FILE ${gcm__FILES})
    include(${gcm__FILE})
  endforeach()
endmacro()

## Copy variables listed in ARGN to the parent scope
macro(up_scope) # ARGN
  foreach(gcm__VAR ${ARGN})
    set(${gcm__VAR} "${${gcm__VAR}}" PARENT_SCOPE)
  endforeach()
endmacro()



#
# Build target generators
#

## Run a target's config scripts and find its sources
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

## Add a library defined by an entry in src/lib
#
# Library targets are distinguished with prefix 'lib' and are written
# to build subdirectory lib.  On Linux systems, shared libraries reuse
# the project version extracted from the project root directory.
function(GenCMake_add_library LIB_PATH)
  GenCMake_configure(${LIB_PATH} LIB_NAME SOURCES)
 
  verbose(STATUS "GenCMake:   Found library 'lib${LIB_NAME}'")
  #  if(CMAKE_COMPILER_IS_GNUCXX)
  #  append(CMAKE_CXX_FLAGS ${GenCMake_CXX_Flags})
    #message(STATUS ${CMAKE_CXX_FLAGS})
  #endif()
  add_library(lib${LIB_NAME} ${SOURCES})
  set_target_properties(lib${LIB_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${GenCMake_LIB_OUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${GenCMake_LIB_OUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${GenCMake_LIB_OUT_DIR}
    OUTPUT_NAME ${LIB_NAME} VERSION ${GenCMake_VERSION})

  target_link_libraries(lib${LIB_NAME} ${GenCMake_LIBRARIES})

endfunction()

## Add a python module defined by an entry in src/python
#
# Python modules are written
# to build subdirectory lib.  On Linux systems, shared libraries reuse
# the project version extracted from the project root directory.
function(GenCMake_add_python PYTHON_PATH)
  GenCMake_configure(${PYTHON_PATH} PYTHON_NAME SOURCES)

  verbose(STATUS "GenCMake:   Found python module '${PYTHON_NAME}'")

  add_library(python-${PYTHON_NAME} ${SOURCES})

  set_target_properties(python-${PYTHON_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${GenCMake_LIB_OUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${GenCMake_LIB_OUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${GenCMake_LIB_OUT_DIR}
    OUTPUT_NAME ${PYTHON_NAME} )

  target_link_libraries(python-${PYTHON_NAME} ${GenCMake_LIBRARIES})
endfunction()

## Add an executable defined by an entry in src/bin
#
# Executable targets reuse the source path's basename and are written
# to build subdirectory bin.  Note that included scripts should fill
# GenCMake_LIBRARIES as appropriate.
function(GenCMake_add_executable BIN_PATH)
  GenCMake_configure(${BIN_PATH} BIN_NAME SOURCES)

  verbose(STATUS "GenCMake:   Found executable '${BIN_NAME}'")

  add_executable(${BIN_NAME} ${SOURCES})

  set_target_properties(${BIN_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${GenCMake_BIN_OUT_DIR})

  target_link_libraries(${BIN_NAME} ${GenCMake_LIBRARIES})
endfunction()

## Add a unit test defined by an entry in src/test
#
# Unit tests are distinguished with suffix '-test' and written to
# build subdirectory test.  Unit tests are not compiled by default
# target 'all' but instead with special target 'all-tests'.  CTest
# runs unit tests with command-line arguments GenCMake_TEST_ARGS.
function(GenCMake_add_test TEST_PATH)
  GenCMake_configure(${TEST_PATH} TEST_NAME SOURCES)

  verbose(STATUS "GenCMake:   Found unit test '${TEST_NAME}'")

  add_executable(${TEST_NAME}-test EXCLUDE_FROM_ALL ${SOURCES})
  add_dependencies(all-tests ${TEST_NAME}-test)

  set_target_properties(${TEST_NAME}-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${GenCMake_TEST_OUT_DIR})

  target_link_libraries(${TEST_NAME}-test ${GenCMake_LIBRARIES})

  add_test(${TEST_NAME}
    ${GenCMake_TEST_OUT_DIR}/${TEST_NAME}-test ${GenCMake_TEST_ARGS})
endfunction()

## Add all targets of type TARGET_TYPE defined beneath SRC_DIR
function(GenCMake_add TARGET_TYPE TARGET_DIR)
  # Glob for scripts shared by all defined targets
  include_glob(${TARGET_DIR}/${SCRIPT_GLOB})

  glob_exclude(TARGETS "(${IGNORED}|${SCRIPT_PAT})" ${TARGET_DIR}/*)
  foreach(TARGET ${TARGETS})
    # Dispatch to appropriate TARGET_TYPE handler
    if(TARGET_TYPE STREQUAL "LIBS")
      GenCMake_add_library(${TARGET})
    elseif(TARGET_TYPE STREQUAL "BINS")
      GenCMake_add_executable(${TARGET})
    elseif(TARGET_TYPE STREQUAL "TESTS")
      GenCMake_add_test(${TARGET})
    elseif(TARGET_TYPE STREQUAL "PYTHONS")
      GenCMake_add_python(${TARGET})
    else()
      message(WARNING "GenCMake_add unrecognized TARGET_TYPE")
    endif()
  endforeach()
endfunction()



#
# GenCMake parameters
#

message(STATUS "Setting defaults in GenCMAKE")
default(VERBOSE FALSE CACHE BOOL
  "Whether to print additional status messages.")

# File globs and regex patterns
default(SCRIPT_GLOB *.gen.cmake CACHE STRING
  "Used to find/include scripts for linking libraries, etc.")
default(SCRIPT_PAT "[.]gen[.]cmake(/|$)" CACHE STRING
  "Used to find/include scripts for linking libraries, etc.")
default(SRC_GLOBS *.c *.C *.c++ *.cc *.cpp *.cxx CACHE STRING
  "Globs matching source files built by this project.")
if(NOT DEFINED IGNORED)
  set(IGNORED "[/.](CVS|te?mp|bac?k|orig|old|out|tags|TAGS)(/|$)")
  set(IGNORED "(/[.]+[^/.]|/#|/_|~(/|$)|${IGNORED})" CACHE STRING
    "Regex matching files to exclude from installation/packaging.")
endif()
message(STATUS "getting input directories for source")
# Script and source input directories
default(GenCMake_SCRIPTS_DIR . CACHE PATH
  "Directory for CMake scripts auto-included by all targets.")
default(GenCMake_INCLUDE_DIR ../include CACHE PATH
  "Headers kept separate from other sources for easy packaging.")
default(GenCMake_LIB_SRC_DIR lib CACHE PATH
  "Source directory for automatically configured libraries.")
default(GenCMake_PYTHON_SRC_DIR python CACHE PATH
  "Source directory for automatically configured libraries.")
default(GenCMake_BIN_SRC_DIR bin CACHE PATH
  "Source directory for automatically configured executables.")
default(GenCMake_TEST_SRC_DIR test CACHE PATH
  "Source directory for automatically configured unit tests.")

# Compiled binary output directories
default(GenCMake_LIB_OUT_DIR lib CACHE PATH
  "Output directory for compiled shared/static libraries.")
default(GenCMake_BIN_OUT_DIR bin CACHE PATH
  "Output directory for compiled binary executables.")
default(GenCMake_TEST_OUT_DIR test CACHE PATH
  "Output directory for compiled unit tests.")
default(CMAKE_DEBUG_POSTFIX -dbg CACHE STRING
  "Library name postfix distinguishing debug builds.")

# The project is named by its containing directory
if(NOT DEFINED GenCMake_PROJECT)
  get_filename_component(GenCMake_PROJECT ${CMAKE_SOURCE_DIR} PATH)
  get_filename_component(GenCMake_PROJECT ${GenCMake_PROJECT} NAME)
  set(GenCMake_PROJECT ${GenCMake_PROJECT} CACHE STRING
    "ID formatted '<name>-<version>' of project to configure.")
endif()

# The version is the name's last dash-delimited token
if(NOT DEFINED GenCMake_VERSION)
  string(REGEX MATCH "[^-]+$" GenCMake_VERSION ${GenCMake_PROJECT})
  set(GenCMake_VERSION ${GenCMake_VERSION} CACHE STRING
    "Version used to distinguish shared libraries.")
endif()

# The build type is normalized for case insensitivity
default(CMAKE_BUILD_TYPE relwithdebinfo CACHE STRING
  "Major compilation mode, e.g. 'release' or 'debug'.")
default(GenCMake_BUILD ${CMAKE_BUILD_TYPE} CACHE STRING
  "The build type used in binary package filenames.")
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE)

# These are later set by auto-included CMake scripts
default(GenCMake_LIBRARIES CACHE STRING
  "Libraries that should link with all compilation targets.")
default(GenCMake_CXX_Flags CACHE STRING
  "Flags thet will be used for the compiler.")
default(GenCMake_TEST_ARGS CACHE STRING
  "Command-line arguments passed when running unit tests.")

# GenCMake operations to perform; set before including
default(GenCMake_DECLARE_PROJECT TRUE)
default(GenCMake_DECLARE_TARGETS TRUE)
default(GenCMake_CONFIG_INSTALL TRUE)
default(GenCMake_CONFIG_PACKAGE TRUE)



#
# Project declaration
#

if(GenCMake_DECLARE_PROJECT)
  verbose(STATUS "GenCMake: Declaring project ${GenCMake_PROJECT}")

  if(CMAKE_BUILD_TYPE STREQUAL "GENERAL")
    # Define 'general' build type as 'relwithdebinfo'
    set(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  elseif(CMAKE_BUILD_TYPE STREQUAL "MINSIZE")
    # Define 'minsize' build type as 'minsizerel'
    set(CMAKE_BUILD_TYPE "MINSIZEREL")
  elseif(CMAKE_BUILD_TYPE STREQUAL "EFENCE")
    # Define 'efence' as 'debug' with Electric Fence
    find_library(ElectricFence_LIBRARY NAMES efence duma)
    list(APPEND GenCMake_LIBRARIES ${ElectricFence_LIBRARY})
    set(CMAKE_BUILD_TYPE "DEBUG")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    append(CMAKE_CXX_FLAGS "-DDEBUG ")
  endif()

  verbose(STATUS "GenCMake:   Build type is '${CMAKE_BUILD_TYPE}'")

  project(${GenCMake_PROJECT} CXX C Fortran)

  include_directories(
    ${GenCMake_INCLUDE_DIR}
    ${GenCMake_LIB_SRC_DIR})

  default(BUILD_SHARED_LIBS FALSE CACHE BOOL
    "Whether to build shared or static libraries.")
  # Enable reasonable warnings on reasonable compilers
  if(CMAKE_COMPILER_IS_GNUCXX)
    append(CMAKE_CXX_FLAGS " -Wall -Wno-sign-compare -fno-strict-aliasing -pthread ")
    if(POSITION_INDEPENDENT) 
      append(CMAKE_CXX_FLAGS "-fPIC")
    endif()
  endif()

  enable_testing()
endif()



#
# Automatic build target generation
#
if(GenCMake_DECLARE_TARGETS)
  verbose(STATUS "GenCMake: Generating build targets")

  add_custom_target(all-tests)

  include_glob(${GenCMake_SCRIPTS_DIR}/${SCRIPT_GLOB})

  GenCMake_add(LIBS ${GenCMake_LIB_SRC_DIR})
  GenCMake_add(PYTHONS ${GenCMake_PYTHON_SRC_DIR})
  GenCMake_add(BINS ${GenCMake_BIN_SRC_DIR})
  GenCMake_add(TESTS ${GenCMake_TEST_SRC_DIR})

endif()



#
# Install configuration
#

#if(GenCMake_CONFIG_INSTALL)
#  verbose(STATUS "GenCMake: Configuring installation")
#
#  install(DIRECTORY
#    ${CMAKE_BINARY_DIR}/${GenCMake_LIB_OUT_DIR}
#    ${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}
#    ${CMAKE_SOURCE_DIR}/${GenCMake_INCLUDE_DIR}
#    DESTINATION . REGEX "${IGNORED}" EXCLUDE)
#endif()
#


#
# Package configuration
#

#if(GenCMake_CONFIG_PACKAGE)
#  verbose(STATUS "GenCMake: Configuring packages")
#
#  default(CPACK_GENERATOR TGZ CACHE STRING
#    "Default package type generated by CPack.")
#  default(CPACK_SOURCE_GENERATOR ${CPACK_GENERATOR})
#
#  # Distinguish compilation packages by platform and build type
#  string(REPLACE "" "" CPACK_PACKAGE_FILE_NAME ${GenCMake_PROJECT}-
#    ${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}-${GenCMake_BUILD})
#  set(CPACK_SOURCE_PACKAGE_FILE_NAME ${GenCMake_PROJECT}-source)
#
#  # Source package contains all source but no builds or packages
#  set(CPACK_SOURCE_INSTALLED_DIRECTORIES ${CMAKE_SOURCE_DIR}/.. .)
#  set(CPACK_SOURCE_IGNORE_FILES "${IGNORED}" "/${GenCMake_PROJECT}-"
#    "/cmake_install" "/CMakeCache" "/CMakeFiles" "[.]build(/|$)")
#
#  include(CPack)
#endif()

