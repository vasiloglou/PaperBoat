INCLUDE(InstallRequiredSystemLibraries)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set ( ARCH64BIT 1 )
else ()
  set ( ARCH64BIT 0 )
endif ()
message ("-- 64bit architecture is ${ARCH64BIT}")
if ( ${ARCH64BIT} EQUAL 1 )
  SET ( CPACK_PACKAGE_ARCHITECTURE "amd64")
else( ${ARCH64BIT} EQUAL 1 )
  SET ( CPACK_PACKAGE_ARCHITECTURE "i386")
endif ( ${ARCH64BIT} EQUAL 1 )


SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The PaperBoat Library")
SET(CPACK_PACKAGE_VENDOR "Ismion Inc.")
SET(CPACK_PACKAGE_CONTACT " Ismion <nvasil@ismion.com>")
SET(CPACK_SOURCE_GENERATOR TGZ)
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "6")
SET(CPACK_PACKAGE_VERSION_PATCH "4")
SET(CPACK_PACKAGE_NAME ismion-paperboat-${CPACK_PACKAGE_ARCHITECTURE})

find_program(BUILD_DEB dpkg)
if (BUILD_DEB)
  message(STATUS "DEB generator found")
  SET(CPACK_GENERATOR  DEB)
endif(BUILD_DEB)


find_program(BUILD_RPM rpmbuild)
if (BUILD_RPM AND NOT BUILD_DEB)
  message(STATUS "RPM generator found")
  SET(CPACK_GENERATOR RPM)
endif(BUILD_RPM AND NOT BUILD_DEB)

  
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Nikolaos Vasiloglou <nvasil@ismion.com>")
SET(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Binary executables for the ismion paperboat library")
SET(CPACK_DEBIAN_PACKAGE_EXTENDED_DESCRIPTION "Binary executabled for running the paperboat algorithms")
SET(CPACK_DEBIAN_PACKAGE_NAME "ismion-paperboat")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS 
  "libboost-all-dev, liblapack-dev, g++, gfortran, libtrilinos-dev")

SET(CPACK_RPM_PACKAGE_ARCHITECTURE ${CPACK_PACKAGE_ARCHITECTURE})
SET(CPACK_RPM_PACKAGE_RELEASE 1)
SET(CPACK_RPM_PACKAGE_LICENSE "ismion")
SET(CPACK_RPM_PACKAGE_URL "http://www.ismion.com")
SET(CPACK_RPM_COMPRESSION_TYPE "gzip")
SET(CPACK_RPM_PACKAGE_REQUIRES "libboost-all-dev,liblapack-dev,gfortran,libtrilinos-dev")


SET(CPACK_SOURCE_IGNORE_FILES
        "*~"
        "*.swp"
        "*.swo"
        "*.log"
        "a.*"
        "temp"
        )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/allkn
  DESTINATION bin/paperboat )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/ams
  DESTINATION bin/paperboat )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/ensvd
  DESTINATION bin/paperboat )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/graphd
  DESTINATION bin/paperboat )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/kde
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/kmeans
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/lasso
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/moe
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/mtsp
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/svd
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/lasso
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/nmf
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/orthogonal_range_search
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/kernel_pca
  DESTINATION bin/paperboat  )
install(PROGRAMS /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/tf3
  DESTINATION bin/paperboat  )

INCLUDE(CPack)
