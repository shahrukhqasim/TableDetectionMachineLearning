cmake_minimum_required(VERSION 3.7)
project(TablesMachineLearning)
link_directories(/usr/local/lib/ /usr/lib/ ${Boost_LIBRARY_DIR})
include_directories(/usr/local/include/ DocProc/src/ ${Boost_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/helpers.cc src/helpers.h)
add_executable(TablesMachineLearning src/main.cc ${SOURCE_FILES})
add_executable(TablesMachineLearningPost src/main_post.cc ${SOURCE_FILES})

target_link_libraries(TablesMachineLearning lept tesseract opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs pthread gomp ${Boost_LIBRARIES} docproc)
target_link_libraries(TablesMachineLearningPost lept tesseract opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs pthread gomp ${Boost_LIBRARIES} docproc)