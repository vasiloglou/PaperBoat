INCLUDE(InstallRequiredSystemLibraries)
file(GLOB libmlpack_dev /.${CMAKE_BINARY_DIR}/${GenCMake_LIB_OUT_DIR}/libmlpack-dev*)
install(FILES  ${libmlpack_dev}
  DESTINATION lib)
file(GLOB libfastlib /.${CMAKE_BINARY_DIR}/${GenCMake_LIB_OUT_DIR}/libfastlib*)
install(FILES ${libfastlib}
  DESTINATION lib)
file(GLOB libubigraph /.${CMAKE_BINARY_DIR}/${GenCMake_LIB_OUT_DIR}/libubigraph*)
install(FILES ${libubigraph}
   DESTINATION lib)
install(FILES ../include/fastlib/base/base.h
  DESTINATION include)
install(FILES ../include/fastlib/base/basic_types.h
  DESTINATION include)
install(FILES ../include/fastlib/base/constant_strings.h
  DESTINATION include)
install(FILES ../include/fastlib/base/logger.h
  DESTINATION include)
install(FILES ../include/fastlib/base/null_stream.h
  DESTINATION include)
file(GLOB external_files /.${CMAKE_BINARY_DIR}/../include/fastlib/external/*.h)
foreach(FILE ${external_files})
  install(FILES /.${FILE}
    DESTINATION include)
endforeach()
file(GLOB external_files /.${CMAKE_BINARY_DIR}/../include/mlpack/external/*.h)
foreach(FILE ${external_files})
  install(FILES /.${FILE}
    DESTINATION include)
endforeach()


SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Paper Boat")
SET(CPACK_PACKAGE_VENDOR "Ismion Inc.")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "6")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_PACKAGE_NAME MLPACK-DEV)
SET(CPACK_ZIP_COMPONENT_INSTALL ON)

INCLUDE(CPack)
