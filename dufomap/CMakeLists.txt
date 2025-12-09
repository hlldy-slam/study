cmake_minimum_required(VERSION 3.17)

project(dufomap
	VERSION 1.0.0
	DESCRIPTION "DUFOMap: Efficient Dynamic Awareness Mapping"
	LANGUAGES CXX
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
		STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(SRC_LIST "${PROJECT_SOURCE_DIR}/src/map/io.cpp")

find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)
#pkg_check_modules(LZF REQUIRED liblzf)
#find_package(liblzf REQUIRED)
			

find_package(OpenMP)
add_library(Map SHARED ${SRC_LIST} ${HEADER_LIST})
add_library(UFO::Map ALIAS Map)

set_target_properties(Map
	PROPERTIES
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_VERSION_MAJOR}
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED YES
	CXX_EXTENSIONS NO
)

# add liblzf
include_directories(include ${PROJECT_SOURCE_DIR}/liblzf-master/)
add_library(liblzf STATIC IMPORTED) 
set_target_properties(Map
                       PROPERTIES IMPORTED_LOCATION
                       /${PROJECT_SOURCE_DIR}/liblzf.a
                       )

# add liblaszip
include_directories(include ${PROJECT_SOURCE_DIR}/include/laszip)
add_library(liblaszip SHARED IMPORTED) 
set_target_properties(liblaszip
                       PROPERTIES IMPORTED_LOCATION
                       ${PROJECT_SOURCE_DIR}/lib/liblaszip.so
                       )

# add libLASlib
include_directories(include ${PROJECT_SOURCE_DIR}/include/lastool_lasLib)
include_directories(include ${PROJECT_SOURCE_DIR}/include/lastool_laszip)
add_library(liblaslib SHARED IMPORTED) 
set_target_properties(liblaslib
                       PROPERTIES IMPORTED_LOCATION
                       ${PROJECT_SOURCE_DIR}/lib/libLASlib.so
                       )

# opencv
find_package(OpenCV 3.4.11 QUIET)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib CACHE PATH "Root directory of tensorflow")
# set(TENSORFLOW_ROOT_DIR lib CACHE PATH "Root directory of tensorflow")

if(NOT OpenCV_FOUND)
	message("warning: cannot find opencv 3.4.11 in system")
	message("         aoto use opencv lib path:" ${LIB_DIR}/libopencv_world.so.3.4)
	message("         if build failed: you can try the flow commond to install opencv dependence: ")
	message("              sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev")

	add_library(libopencv SHARED IMPORTED) 
	set_target_properties(libopencv
                       PROPERTIES IMPORTED_LOCATION
                       ${LIB_DIR}/libopencv_world.so.3.4
                       )
  	set(OpenCV_LIBS libopencv)
endif()

set(OPENCV_INCLUDES ${PROJECT_SOURCE_DIR}/include/opencv)
include_directories(include ${PROJECT_SOURCE_DIR}/include/opencv)

# add downsample
add_subdirectory(${PROJECT_SOURCE_DIR}/src/xxHash)
include_directories(include ${PROJECT_SOURCE_DIR}/src/xxHash)

# pcl
# find_package(PCL REQUIRED)
# include_directories(include /usr/local/include/pcl-1.8/)
# set(PCL_DIR "/opt/WD-1T-1/jyd2/slam_projects/pcl-pcl-1.14.1")
#set(PCL_DIR "/usr/local/include/pcl-1.13")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PCL_DIR}/cmake")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#添加open3d路径
find_package(Open3D REQUIRED)
set(Open3D_DIR "/usr/local/include/Open3D")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Open3D_DIR}/cmake")
include_directories(${Open3D_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS})
add_definitions(${Open3D_DEFINITIONS})

target_include_directories(Map
	PUBLIC
	$<INSTALL_INTERFACE:include>
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_link_libraries(Map
	PUBLIC
	${LZ4_LIBRARIES}
	OpenMP::OpenMP_CXX
)

find_package(TBB QUIET)

IF(TBB_FOUND)
	message(STATUS "TBB found, enabling parallel execution")
	target_compile_definitions(Map
		PUBLIC
		UFO_PARALLEL=1
	)
	target_link_libraries(Map
		PUBLIC
		TBB::tbb
	)
ELSE()
	message(STATUS "TBB not found, disabling parallel execution")
ENDIF()

target_compile_features(Map
	PUBLIC
	cxx_std_20
)

target_compile_options(Map
	PUBLIC
	-fPIC
)

# add_subdirectory(src)

include_directories(include ${PROJECT_SOURCE_DIR}/include/ground_seg)
include_directories(include ${PROJECT_SOURCE_DIR}/include/eigen-3.4.0)

# 
add_executable(dufomap_run src/LASPointReader.cpp src/main.cpp src/clean_map.cpp src/LASPointWriter.cpp src/voxel_grid_downsampling.cpp src/stuff.cpp src/groud_seg.cpp src/ransac.cpp src/patchworkpp.cpp)
set_target_properties(dufomap_run
	PROPERTIES
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_VERSION_MAJOR}
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED YES
	CXX_EXTENSIONS NO
)
target_link_libraries(dufomap_run PRIVATE UFO::Map xxhash liblaszip liblaslib  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Open3D_LIBRARIES} -lstdc++fs)
target_compile_features(dufomap_run
	PUBLIC
	cxx_std_20
)
target_compile_options(dufomap_run
	PRIVATE
	# -Wall
	# -Werror
	# -Wextra
	# -Wpedantic
	# -Wunreachable-code
	# -Wconversion
	# -Wcast-align
	# -Wunused
	# -Wno-unused-parameter
	# -Wold-style-cast
	# -Wpointer-arith
	# -Wcast-qual
	# -Wno-missing-braces
	# -Wdouble-promotion

	# -Wshadow
	# -march=native
)


set(UFOMAP_BMI2 FALSE CACHE BOOL "Enable/disable BMI2 instructions")

if(DEFINED ENV{UFOMAP_BMI2})
	set(UFOMAP_BMI2 $ENV{UFOMAP_BMI2})
endif(DEFINED ENV{UFOMAP_BMI2})

if(UFOMAP_BMI2)
	message(STATUS "UFOMAP BMI2 instructions enabled")
	target_compile_options(Map
		PUBLIC
		-mbmi2
	)
else()
	message(STATUS "UFOMAP BMI2 instructions disabled")
endif(UFOMAP_BMI2)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})

install(TARGETS Map
	EXPORT ufomapTargets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT ufomapTargets
	FILE "ufomapTargets.cmake"
	NAMESPACE UFO::
	DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
	"${PROJECT_SOURCE_DIR}/cmake/ufomapConfig.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/ufomapConfig.cmake"
	INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/ufomapConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

INSTALL(
	FILES
	"${CMAKE_CURRENT_BINARY_DIR}/ufomapConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/ufomapConfigVersion.cmake"
	DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/include/
	DESTINATION include
)
