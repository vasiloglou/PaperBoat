INCLUDE(FindThreads)
list(APPEND GenCMake_LIBRARIES
   ${CMAKE_THREAD_LIBS_INIT} ) 
