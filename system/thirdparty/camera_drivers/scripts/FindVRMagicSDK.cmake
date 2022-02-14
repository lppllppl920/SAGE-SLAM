include(FindPackageHandleStandardArgs)

find_path(VRMAGICSDK_INCLUDE_DIRS1 NAMES vrmusbcam2.h PATHS /opt/vrmagic/sdk-4.3.0/x64/development_kit/include)
find_path(VRMAGICSDK_INCLUDE_DIRS2 NAMES vrmusbcamcpp.h PATHS /opt/vrmagic/sdk-4.3.0/x64/development_kit/wrappers/c++)
find_file(VRMAGICSDK_CPP_WRAPPER NAMES vrmusbcamcpp.cpp PATHS /opt/vrmagic/sdk-4.3.0/x64/development_kit/wrappers/c++)
find_library(VRMAGICSDK_LIBRARIES NAMES vrmusbcam2 PATHS /opt/vrmagic/sdk-4.3.0/x64/development_kit/lib)

find_package_handle_standard_args(VRMagicSDK "Could NOT find VRMagic SDK." VRMAGICSDK_LIBRARIES VRMAGICSDK_INCLUDE_DIRS1 VRMAGICSDK_INCLUDE_DIRS2 VRMAGICSDK_CPP_WRAPPER)

if(VRMAGICSDK_FOUND)
    find_package_message(VRMAGICSDK_FOUND "Found VRMagic SDK" "[${VRMAGICSDK_LIBRARIES}][${VRMAGICSDK_INCLUDE_DIRS}]")
    set(VRMAGICSDK_INCLUDE_DIRS ${VRMAGICSDK_INCLUDE_DIRS1} ${VRMAGICSDK_INCLUDE_DIRS2})
endif(VRMAGICSDK_FOUND)

mark_as_advanced(VRMAGICSDK_INCLUDE_DIRS VRMAGICSDK_INCLUDE_DIRS1 VRMAGICSDK_INCLUDE_DIRS2 VRMAGICSDK_LIBRARIES VRMAGICSDK_CPP_WRAPPER)
