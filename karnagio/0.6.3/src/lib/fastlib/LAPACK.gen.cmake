SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
INCLUDE(FindLAPACK)
#INCLUDE(FindBLAS)

#find_library(BLAS_LIBRARY NAMES blas-3 blas)
#find_library(LAPACK_LIBRARY NAMES lapack-3 lapack)
#STRING(REPLACE ".so" ".a" LAPACK_LIBRARY ${LAPACK_LIBRARY})
#STRING(REPLACE ".so" ".a" BLAS_LIBRARY ${BLAS_LIBRARY})
#list(APPEND GenCMake_LIBRARIES ${LAPACK_LIBRARY} ${BLAS_LIBRARY})
list(APPEND GenCMake_LIBRARIES ${LAPACK_LIBRARIES})

#STRING(REPLACE ".so" ".a" LAPACK_LIBRARY ${LAPACK_LIBRARY})
#STRING(REPLACE ".so" ".a" BLAS_LIBRARY ${BLAS_LIBRARY})
#list(APPEND GenCMake_LIBRARIES
#   ${LAPACK_LIBRARY} ) 
#list(APPEND GenCMake_LIBRARIES
#  ${BLAS_LIBRARY})
list(APPEND GenCMake_LIBRARIES
     -Wl,-Bstatic gfortran -Wl,-Bdynamic)
#message(${LAPACK_LIBRARY})
#message(${BLAS_LIBRARY})
