include(FindPackageHandleStandardArgs)

find_path(FLYCAPTURE2_INCLUDE_DIR FlyCapture2.h PATHS /opt/flycapture/include/flycapture /usr/include/flycapture)
find_library(FLYCAPTURE2_LIBRARY NAMES flycapture PATHS /opt/flycapture/lib /usr/lib /usr/lib32 /usr/lib64)
find_library(FLYCAPTURE2_GUI_LIBRARY NAMES flycapturegui PATHS /opt/flycapture/lib /usr/lib /usr/lib32 /usr/lib64)

find_package_handle_standard_args(FlyCapture2 "Could NOT find Flycapture SDK." FLYCAPTURE2_LIBRARY FLYCAPTURE2_INCLUDE_DIR)

if(FLYCAPTURE2_FOUND)
    find_package_message(FLYCAPTURE2_FOUND "Found Fly Capture SDK  ${FLYCAPTURE2_LIBRARY}" "[${FLYCAPTURE2_LIBRARY}][${FLYCAPTURE2_INCLUDE_DIR}]")
endif(FLYCAPTURE2_FOUND)

mark_as_advanced(FLYCAPTURE2_INCLUDE_DIR FLYCAPTURE2_LIBRARY FLYCAPTURE2_GUI_LIBRARY)
