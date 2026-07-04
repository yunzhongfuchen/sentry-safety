# Install script for directory: /workspace/sophon-stream-master

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/workspace/sophon-stream-master/build/framework/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/posec3d/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/fastpose/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/yolox/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/yolov5/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/yolov7/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/bytetrack/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/resnet/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/openpose/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/retinaface/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/lprnet/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/ppocr/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/yolov8/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/algorithm/lightstereo/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/multimedia/decode/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/multimedia/osd/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/blank/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/distributor/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/converger/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/http_push/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/faiss/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/dwa/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/dpu/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/blend/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/ive/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/stitch/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/resize/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/filter/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/element/tools/qt_display/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/3rdparty/freetype2/cmake_install.cmake")
  include("/workspace/sophon-stream-master/build/samples/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/workspace/sophon-stream-master/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
