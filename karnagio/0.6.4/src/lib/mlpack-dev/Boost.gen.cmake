set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost COMPONENTS program_options serialization system)

list(APPEND GenCMake_LIBRARIES ${Boost_LIBRARIES})

include_directories(${Boost_INCLUDE_DIRS})

#add_definitions(-DBOOST_ALL_DYN_LINK)
