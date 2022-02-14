include(FindPackageHandleStandardArgs)

find_path(IDSUEYE_INCLUDE_DIR ueye.h PATHS /usr/include)
find_library(IDSUEYE_LIBRARY NAMES ueye_api PATHS /usr/lib64 /usr/lib /usr/lib32)

find_package_handle_standard_args(IDSuEye "Could NOT find IDS uEye." IDSUEYE_LIBRARY IDSUEYE_INCLUDE_DIR)

if(IDSUEYE_FOUND)
    find_package_message(IDSUEYE_FOUND "Found IDS uEyE  ${IDSUEYE_LIBRARY}" "[${IDSUEYE_LIBRARY}][${IDSUEYE_INCLUDE_DIR}]")
endif(IDSUEYE_FOUND)

mark_as_advanced(IDSUEYE_INCLUDE_DIR IDSUEYE_LIBRARY)
