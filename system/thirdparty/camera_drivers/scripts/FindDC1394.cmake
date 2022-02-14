#
# try to find the dc1394 library (version 2) and include files
#
# DC1394_INCLUDE_DIR, where to find dc1394/dc1394_control.h, etc.
# DC1394_LIBRARIES, the libraries to link against to use DC1394.
# DC1394_FOUND, If false, do not try to use DC1394.
#


find_path( DC1394_INCLUDE_DIR dc1394/control.h
  /usr/include
  /usr/local/include
)

find_library( DC1394_LIBRARIES dc1394
  /usr/lib64
  /usr/lib
  /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DC1394 DEFAULT_MSG DC1394_INCLUDE_DIR DC1394_LIBRARIES)

mark_as_advanced(DC1394_INCLUDE_DIR DC1394_LIBRARIES)
