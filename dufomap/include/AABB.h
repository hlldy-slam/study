

#ifndef AABB_H
#define AABB_H


#include <math.h>
#include <algorithm>

#include "Vector3.h"

using std::min;
using std::max;
using std::endl;

namespace Potree{

class AABB{

public:
	Vector3<double> min;
	Vector3<double> max;
	Vector3<double> size;

	AABB(){
		min = Vector3<double>(std::numeric_limits<float>::max());
		max = Vector3<double>(-std::numeric_limits<float>::max());
		size = Vector3<double>(std::numeric_limits<float>::max());
	}

	AABB(Vector3<double> min, Vector3<double> max){
		this->min = min;
		this->max = max;
		size = max-min;
	}

	bool isInside(const Vector3<double> &p){
		if(min.x <= p.x && p.x <= max.x){
			if(min.y <= p.y && p.y <= max.y){
				if(min.z <= p.z && p.z <= max.z){
					return true;
				}
			}
		}

		return false;
	}

	// add jyd 20210804
	bool isInside(double x, double y, double z, double tr[16]) {
		/*double t_x = tr[0] * x + tr[4] * y + tr[8] * z + tr[12];
		double t_y = tr[1] * x + tr[5] * y + tr[9] * z + tr[13];
		double t_z = tr[2] * x + tr[6] * y + tr[10] * z + tr[14];*/

		double t_x = tr[0] * x + tr[1] * y + tr[2] * z + tr[3];
		double t_y = tr[4] * x + tr[5] * y + tr[6] * z + tr[7];
		double t_z = tr[8] * x + tr[9] * y + tr[10] * z + tr[11];
	
		if (min.x <= t_x && t_x <= max.x) {
			if (min.y <= t_y && t_y <= max.y) {
				if (min.z <= t_z && t_z <= max.z) {
					return true;
				}
			}
		}

		return false;
	}

	bool isInside( Vector3<double>& p, double tr[16]) {
		/*p.x = tr[0] * p.x + tr[4] * p.y + tr[8] * p.z + tr[12];
		p.y = tr[1] * p.x + tr[5] * p.y + tr[9] * p.z + tr[13];
		p.z = tr[2] * p.x + tr[6] * p.y + tr[10] * p.z + tr[14];*/

		p.x = tr[0] * p.x + tr[1] * p.y + tr[2] * p.z + tr[3];
		p.y = tr[4] * p.x + tr[5] * p.y + tr[6] * p.z + tr[7];
		p.z = tr[8] * p.x + tr[9] * p.y + tr[10] * p.z + tr[11];

		if (min.x <= p.x && p.x <= max.x) {
			if (min.y <= p.y && p.y <= max.y) {
				if (min.z <= p.z && p.z <= max.z) {
					return true;
				}
			}
		}

		return false;
	}

	void update(const Vector3<double> &point){
		min.x = std::min(min.x, point.x);
		min.y = std::min(min.y, point.y);
		min.z = std::min(min.z, point.z);

		max.x = std::max(max.x, point.x);
		max.y = std::max(max.y, point.y);
		max.z = std::max(max.z, point.z);

		size = max - min;
	}

	void update(const AABB &aabb){
		update(aabb.min);
		update(aabb.max);
	}

	void makeCubic(){
		max = min + size.maxValue();
		size = max - min;
	}

	friend ostream &operator<<( ostream &output,  const AABB &value ){ 
		output << "min: " << value.min << endl;
		output << "max: " << value.max << endl;
		output << "size: " << value.size << endl;
		return output;            
	}

};

}

#endif