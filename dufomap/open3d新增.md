1.cmakelist 第八行新增：set(CMAKE_POSITION_INDEPENDENT_CODE ON)

2.cmakelist 104-110行：
	#添加open3d路径
	find_package(Open3D REQUIRED)
	set(Open3D_DIR "/usr/local/include/Open3D")
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Open3D_DIR}/cmake")
	include_directories(${Open3D_INCLUDE_DIRS})
	link_directories(${Open3D_LIBRARY_DIRS})
	add_definitions(${Open3D_DEFINITIONS})
	
3.clean_map.h文件第五行添加绝对路径的Open3D.h头文件：
	//0.9.0版Open3D的绝对路径头文件
	#include </usr/local/include/Open3D/Open3D.h>
	

