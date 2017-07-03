cmake_minimum_required(VERSION 3.7)
project(TablesMachineLearning)
link_directories(/usr/local/lib/ /usr/lib/ ${Boost_LIBRARY_DIR})
include_directories(/usr/local/include/ DocProc/src/ ${Boost_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp)
add_executable(TablesMachineLearning ${SOURCE_FILES})

target_link_libraries(TablesMachineLearning lept tesseract opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs pthread gomp ${Boost_LIBRARIES} docproc)