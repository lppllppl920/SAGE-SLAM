set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ------------------------------------------------------------------------------
# List sources
# ------------------------------------------------------------------------------
set(core_sources
  # gtsam
  gtsam/geometric_factor.cpp
  gtsam/photometric_factor.cpp
  gtsam/reprojection_factor.cpp
  gtsam/match_geometry_factor.cpp
  gtsam/loop_mg_factor.cpp
  gtsam/scale_factor.cpp
  gtsam/pose_factor.cpp
  gtsam/code_factor.cpp
  gtsam/rel_pose_factor.cpp
  gtsam/rel_pose_scale_factor.cpp
  # system
  system/camera_tracker.cpp
  system/dl_descriptor.cpp
  system/loop_detector.cpp
  system/tensor_vocabulary.cpp
  # mapping
  mapping/df_work.cpp
  mapping/mapper.cpp
  mapping/mapping_utils.cpp
  mapping/work_manager.cpp
  mapping/work.cpp
  # network
  network/code_depth_network.cpp
  network/feature_network.cpp
  # top-level
  deepfactors_options.cpp
  deepfactors.cpp
)

set(core_headers
  # gtsam
  gtsam/geometric_factor.h
  gtsam/gtsam_traits.h
  gtsam/gtsam_utils.h
  gtsam/photometric_factor.h
  gtsam/reprojection_factor.h
  gtsam/match_geometry_factor.h
  gtsam/loop_mg_factor.h
  gtsam/scale_factor.h
  gtsam/pose_factor.h
  gtsam/code_factor.h
  gtsam/rel_pose_factor.h
  gtsam/rel_pose_scale_factor.h
  # system
  system/camera_tracker.h
  system/dl_descriptor.h
  system/loop_detector.h
  system/tensor_vocabulary.h
  # mapping
  mapping/df_work.h
  mapping/frame.h
  mapping/keyframe_map.h
  mapping/keyframe.h
  mapping/mapper.h
  mapping/mapping_utils.h
  mapping/work_manager.h
  mapping/work.h
  # network
  network/code_depth_network.h
  network/feature_network.h
  # top-level
  deepfactors_options.h
  deepfactors.h
)

# ------------------------------------------------------------------------------
# Build target
# ------------------------------------------------------------------------------
add_library(df_core SHARED ${core_sources} ${core_headers})
target_include_directories(df_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/gtsam
  ${CMAKE_CURRENT_SOURCE_DIR}/mapping
  ${CMAKE_CURRENT_SOURCE_DIR}/network
  ${CMAKE_CURRENT_SOURCE_DIR}/system
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(df_core 
  PUBLIC
  df_common
  df_cuda
  Eigen3::Eigen
  ${DBoW2_LIBRARIES}
  gtsam
  opengv
  ${glog_LIBRARIES}
  gflags
  teaserpp::teaser_registration
  teaserpp::teaser_io
  ${TORCH_LIBRARIES}
  ${HDF5_CXX_LIBRARIES}
  ${HDF5_LIBRARIES}
)

