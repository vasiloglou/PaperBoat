INCLUDE(InstallRequiredSystemLibraries)
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PaperBoat Karnagio")
SET(CPACK_PACKAGE_VENDOR "Ismion Inc.")
SET(CPACK_PACKAGE_CONTACT " Ismion <nvasil@ismion.com>")
SET(CPACK_SOURCE_GENERATOR TGZ)
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "6")
SET(CPACK_PACKAGE_VERSION_PATCH "3")
SET(CPACK_PACKAGE_NAME paperboat-dev)
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_NAME "paperboat")

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set ( ARCH64BIT 1 )
else ()
  set ( ARCH64BIT 0 )
endif ()
message ("-- 64bit architecture is ${ARCH64BIT}")
if ( ${ARCH64BIT} EQUAL 1 )
  SET ( CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
else( ${ARCH64BIT} EQUAL 1 )
  SET ( CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i686")
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

install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/allkn
  DESTINATION /opt/ismion/paperboat/)
install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/kde
  DESTINATION /opt/ismion/paperboat/)
install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/kmeans
  DESTINATION /opt/ismion/paperboat/)
install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/lasso
  DESTINATION /opt/ismion/paperboat/)
install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/regression
  DESTINATION /opt/ismion/paperboat/)
install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/svd
  DESTINATION /opt/ismion/paperboat/)
install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/svm
  DESTINATION /opt/ismion/paperboat/)
install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/nmf
  DESTINATION /opt/ismion/paperboat/)
install(FILES  /.${CMAKE_BINARY_DIR}/${GenCMake_BIN_OUT_DIR}/npr
  DESTINATION /opt/ismion/paperboat/)


include (CPack)






