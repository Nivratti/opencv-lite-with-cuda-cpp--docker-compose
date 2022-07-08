cmake_minimum_required(VERSION 3.9)

project (opencv_dnn_static_build_demo)

add_definitions(-std=c++11)
add_definitions("-Wall")

set( CMAKE_CXX_FLAGS " -pthread -lnppc" )

# =============================================================================
# make executable for testing purpose
# -----------------------------------------------------------------------------
add_executable(main
	main_executable.cpp
)

# Find Package
find_package( OpenCV REQUIRED )

# MESSAGE(STATUS ${OpenCV_LIBS})

if( OpenCV_FOUND )
  # For development first try adding libraries one by one without static
  # Adding all libraries at once and then building may cause issue -- so add libraries one ny one, or comment some code part and build step by step
  # target_link_libraries( main opencv_imgcodecs opencv_dnn opencv_objdetect)

  ## opencv_imgcodecs -- for cv2.imread
  ## opencv_dnn -- for dnn module
  ## opencv_objdetect -- for importing CascadeClassifier
  ## Finally after successful cmake -- add only required libraries
  ## target_link_libraries( main opencv_imgcodecs opencv_dnn opencv_objdetect "-static")
  target_link_libraries( main ${OpenCV_LIBS})
endif()
