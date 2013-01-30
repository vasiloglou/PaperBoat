find_package(Boost COMPONENTS unit_test_framework)
list(APPEND GenCMake_LIBRARIES ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
include_directories(${Boost_INCLUDE_DIRS})
