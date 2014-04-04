set(CMAKE_LIBRARY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../repos/boost_1_55_0/stage/lib)
set(CMAKE_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../repos/boost_1_55_0)
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_USE_MULTITHREADED OFF)


find_package(Boost COMPONENTS thread)

list(APPEND GenCMake_LIBRARIES ${Boost_LIBRARIES} rt)

include_directories(${Boost_INCLUDE_DIRS})

#add_definitions(-DBOOST_ALL_DYN_LINK)
