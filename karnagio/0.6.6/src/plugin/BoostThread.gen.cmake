set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost COMPONENTS thread)

list(APPEND GenCMake_LIBRARIES ${Boost_LIBRARIES} rt)

include_directories(${Boost_INCLUDE_DIRS})

#add_definitions(-DBOOST_ALL_DYN_LINK)
