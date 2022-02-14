# ------------------------------------------------------------------------------
# Find all the cmake-based dependencies
# If you haven't built them, this should also be able to find other installations
# You can point find_package to them by setting e.g. the Sophus_DIR variable
# Note that we're using HINTS instead of PATHS to make sure to search the
# local build first before looking into cmake package registry
# See: https://cmake.org/cmake/help/v3.0/command/find_package.html
# ------------------------------------------------------------------------------
set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install_${CMAKE_BUILD_TYPE}")
set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build_${CMAKE_BUILD_TYPE}")
set(CMAKE_PREFIX_PATH ${INSTALL_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# Add a DBoW2 imported target because it does something weird in its config
# ------------------------------------------------------------------------------
add_library(DBoW2 SHARED IMPORTED GLOBAL)
message(STATUS "DBOW2 library path ${DBoW2_LIBRARY}")
set_property(TARGET DBoW2 APPEND PROPERTY IMPORTED_LOCATION
  ${DBoW2_LIBRARY})
set_property(TARGET DBoW2 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${DBoW2_INCLUDE_DIR})
