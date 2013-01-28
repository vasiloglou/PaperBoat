cmake_minimum_required(VERSION 2.8.8)

set(CMAKE_INCLUDE_PATH /include)
set(CMAKE_LIBRARY_PATH /lib)
function(find_file_in_parents VAR FILENAME DIR)
  find_file(${VAR} ${FILENAME} ${DIR})

  while(NOT ${VAR} AND DIR AND NOT DIR STREQUAL "/" )
    get_filename_component(DIR ${DIR} PATH)
    find_file(${VAR} ${FILENAME} ${DIR})
  endwhile()

  set(${VAR} ${${VAR}} PARENT_SCOPE)
endfunction()

message(STATUS "getting source dir")
get_filename_component(GenCMake_VERSION ${CMAKE_SOURCE_DIR} PATH)
message(STATUS "getting cmake version")
get_filename_component(GenCMake_VERSION ${GenCMake_VERSION} NAME)
set(GenCMake_PROJECT fl-lite-${GenCMake_VERSION})

#find_file_in_parents(DATASETS_DIR datasets ${CMAKE_SOURCE_DIR})
#set(GenCMake_TEST_ARGS "${DATASETS_DIR}")



include(GenCMake.cmake NO_POLICY_SCOPE)
include(CPack.cmake NO_POLICY_SCOPE)
