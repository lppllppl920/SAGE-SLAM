include(CMakeFindDependencyMacro)

if(0)
    find_dependency(cereal QUIET)
endif()

if(0)
    find_dependency(CameraModels QUIET)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/CameraDriversTargets.cmake")

set(CameraDrivers_SUPPORTS_POINTGREY )
set(CameraDrivers_SUPPORTS_FIREWIRE ON)
set(CameraDrivers_SUPPORTS_VRMAGIC )
set(CameraDrivers_SUPPORTS_OPENNI2 ON)
set(CameraDrivers_SUPPORTS_REALSENSE )
set(CameraDrivers_SUPPORTS_REALSENSE2 )
set(CameraDrivers_SUPPORTS_KINECTONE )
set(CameraDrivers_SUPPORTS_IDSUEYE )
set(CameraDrivers_SUPPORTS_V4L2 ON)
