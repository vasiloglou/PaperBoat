set(Boost_USE_STATIC_LIBS  OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS serialization program_options system)
list(APPEND GenCMake_LIBRARIES  ${Boost_LIBRARIES})

include_directories(${Boost_INCLUDE_DIRS})

add_definitions(-DBOOST_ALL_DYN_LINK)