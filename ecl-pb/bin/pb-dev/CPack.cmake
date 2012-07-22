INCLUDE(InstallRequiredSystemLibraries)
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ECL PaperBoat")
SET(CPACK_PACKAGE_VENDOR "Ismion Inc.")
SET(CPACK_PACKAGE_CONTACT " Ismion <nvasil@ismion.com>")
SET(CPACK_SOURCE_GENERATOR TGZ)
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "6")
SET(CPACK_PACKAGE_VERSION_PATCH "3")
SET(CPACK_PACKAGE_NAME ecl-paperboat-dev)
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Nikolaos Vasiloglou <nvasil@ismion.com>")
SET(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Ecl wrapper of the ismion paperboat library")
SET(CPACK_DEBIAN_PACKAGE_EXTENDED_DESCRIPTION "Ecl wrapper fo the ismion paperboat. This is the first level of light integration with the HPCC platform")
SET(CPACK_DEBIAN_PACKAGE_NAME "ecl-paperboat")

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set ( ARCH64BIT 1 )
else ()
  set ( ARCH64BIT 0 )
endif ()
message ("-- 64bit architecture is ${ARCH64BIT}")
if ( ${ARCH64BIT} EQUAL 1 )
  SET ( CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
else( ${ARCH64BIT} EQUAL 1 )
  SET ( CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
endif ( ${ARCH64BIT} EQUAL 1 )

SET(CPACK_DEBIAN_PACKAGE_DEPENDS 
  "libboost-all-dev, liblapack-dev, g++, gfortran, libtrilinos-dev")

SET(CPACK_SOURCE_IGNORE_FILES
        "*~"
        "*.swp"
        "*.swo"
        "*.log"
        "a.*"
        "temp"
        )

install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_LIB_OUT_DIR}/libecl-paperboat.a
  DESTINATION lib/)
install(DIRECTORY ../../karnagio/include
  DESTINATION include/karnagio/)
install(DIRECTORY ../../karnagio/src/lib/fastlib
  DESTINATION include/karnagio/src/fastlib)
install(DIRECTORY ../../karnagio/src/lib/mlpack
  DESTINATION include/karnagio/src/mlpack)
install(DIRECTORY ../../ecl-pb-glue
  DESTINATION include/)


include (CPack)
