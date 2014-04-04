set(CMAKE_LIBRARY_PATH /local/home/nikolaos.vasiloglou/boost_1_54_0/stage/lib)
set(CMAKE_INCLUDE_PATH /local/home/nikolaos.vasiloglou/boost_1_54_0)
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_USE_MULTITHREADED OFF)

find_package(Boost COMPONENTS program_options serialization system filesystem iostreams)

list(APPEND GenCMake_LIBRARIES ${Boost_LIBRARIES})

include_directories(${Boost_INCLUDE_DIRS})
find_package( ZLIB REQUIRED )
if ( ZLIB_FOUND )
  include_directories( ${ZLIB_INCLUDE_DIRS} )
  list(APPEND GenCMake_LIBRARIES ${ZLIB_LIBRARIES})
endif( ZLIB_FOUND )
#add_definitions(-DBOOST_ALL_DYN_LINK)
