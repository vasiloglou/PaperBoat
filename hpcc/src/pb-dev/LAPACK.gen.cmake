INCLUDE(FindLAPACK)
#STRING(REPLACE ".so" ".a" LAPACK_LIBRARIES ${LAPACK_LIBRARIES})
list(APPEND GenCMake_LIBRARIES ${LAPACK_LIBRARIES})
#find_library(BLAS_LIBRARY NAMES blas-3 blas)
#find_library(LAPACK_LIBRARY NAMES lapack-3 lapack)
#STRING(REPLACE ".so" ".a" LAPACK_LIBRARY ${LAPACK_LIBRARY})
#STRING(REPLACE ".so" ".a" BLAS_LIBRARY ${BLAS_LIBRARY})
#list(APPEND GenCMake_LIBRARIES
#   ${LAPACK_LIBRARY} ) 
#list(APPEND GenCMake_LIBRARIES
#  ${BLAS_LIBRARY})
list(APPEND GenCMake_LIBRARIES
     -Wl,-Bstatic gfortran -Wl,-Bdynamic)

#MESSAGE(${GenCMake_LIBRARIES})
