cmake_minimum_required(VERSION 3.0)
project(scale_estimation CXX C)

add_compile_options(-std=c++14)

# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses installed package
include_directories(${GTSAM_INCLUDE_DIR})
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add executables
#add_executable(test_localization src/LocalizationExample.cpp)
#target_link_libraries(test_localization ${GTSAM_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})

set(GTSAM_LIBRARIES gtsam)

find_package(GTSAMCMakeTools)
include(GtsamMakeConfigFile)
include(GtsamBuildTypes)
include(GtsamTesting)

# for unittest scripts
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")

# Add executables
gtsamAddExamplesGlob("src/*.cpp" "" "gtsam")