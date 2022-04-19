# ------------------------------------------------------------------------------
# result_viewer
# Display a ground-truth trajectory and reprojected depth from a dataset and
# an estimated trajectory 
# ------------------------------------------------------------------------------
add_executable(result_viewer result_viewer.cpp)
target_include_directories(result_viewer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(result_viewer PUBLIC
  df_common
  df_core
  df_gui
  df_drivers
  opencv_core
  opencv_highgui
  gflags
  # ncurses
  ncursesw
  # glog
  ${glog_LIBRARIES}
)

# ------------------------------------------------------------------------------
# voc_builder
# ------------------------------------------------------------------------------
add_executable(voc_builder voc_builder.cpp)
target_include_directories(voc_builder PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(voc_builder PUBLIC
  df_common
  df_core
  df_drivers
  opencv_core
  opencv_highgui
  gflags
  ${glog_LIBRARIES}
  ${DBoW2_LIBRARIES}
  stdc++fs
  ${HDF5_CXX_LIBRARIES}
  ${HDF5_LIBRARIES}
)

# # ------------------------------------------------------------------------------
# # voc_test
# # Test a BoW vocabulary by calculating similarity between example images
# # ------------------------------------------------------------------------------
# add_executable(voc_test voc_test.cpp)
# target_include_directories(voc_test PUBLIC
#   ${CMAKE_CURRENT_SOURCE_DIR}
# )
# target_link_libraries(voc_test PUBLIC
#   df_core
#   opencv_core
#   opencv_highgui
#   DBoW2
# )


