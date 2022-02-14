# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(common_sources
  display_utils.cpp
  interp.cpp
  logutils.cpp
  timing.cpp
)

set(common_headers
  camera_pyramid.h
  display_utils.h
  indexed_map.h
  interp.h
  logutils.h
  pinhole_camera_impl.h
  pinhole_camera.h
  timing.h
  tum_io.h
)

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_library(df_common SHARED ${common_sources} ${common_headers})
target_include_directories(df_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(df_common PUBLIC
   Eigen3::Eigen
   Sophus::Sophus
   opencv_core
   opencv_highgui
   opencv_imgproc
   opencv_features2d
   ${TORCH_LIBRARIES}
)
