

#ifndef POINTREADER_H
#define POINTREADER_H

#include "osplatformutil.h"
#if defined I_OS_LINUX
	#include <experimental/filesystem>
	namespace fs = std::experimental::filesystem;
#else
	#include <filesystem>
	namespace fs = std::filesystem;
#endif

#include "Point.h"
#include "AABB.h"

namespace Potree{

class PointReader{
public:

	virtual ~PointReader(){};

	virtual bool readNextPoint() = 0;

	virtual Point getPoint() = 0;

	virtual AABB getAABB() = 0;

	virtual long long numPoints() = 0;

	virtual void close() = 0;
};

}

#endif