#find_library(LAPACK liblapack.a)
#find_library(BLAS libblas.a)
#list(APPEND GenCMake_LIBRARIES ${LAPACK} ${BLAS})
#include(FindLAPACK)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../repos/lapack-3.4.2/ ${CMAKE_CURRENT_BINARY_DIR}/lapack)
list(APPEND GenCMake_LIBRARIES lapack)


#find_library(BLAS_LIBRARY NAMES blas-3 blas)
#find_library(LAPACK_LIBRARY NAMES lapack-3 lapack)
#STRING(REPLACE ".so" ".a" LAPACK_LIBRARY ${LAPACK_LIBRARY})
#STRING(REPLACE ".so" ".a" BLAS_LIBRARY ${BLAS_LIBRARY})
#list(APPEND GenCMake_LIBRARIES
#   ${LAPACK_LIBRARY} ) 
#list(APPEND GenCMake_LIBRARIES
#  ${BLAS_LIBRARY})
#list(APPEND GenCMake_LIBRARIES)
#     -Wl,-Bstatic gfortran -Wl,-Bdynamic)

#MESSAGE(${GenCMake_LIBRARIES})
