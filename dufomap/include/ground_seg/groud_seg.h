#pragma once
#include <vector>
#include <iostream>

#define PI   3.1415926535897932384626433832795

class Groud_seg
{
public:
	Groud_seg();
	~Groud_seg();

	
	void groud_seg_my(std::vector<float>  &cloud,  std::vector<float>  &out_ground_cloud, std::vector<float>  &out_non_ground_cloud, std::string out_ply_path="");
	void groud_seg_patchworkpp(std::vector<float> &cloud, std::string out_path);

	bool groud_seg_writePLY_ascii(const std::string& filename,
			std::vector<float>  &cloud_ground,
			std::vector<unsigned char>&ground_color,
			std::vector<float>  &non_cloud_ground,
			std::vector<unsigned char>&non_ground_color);

private:
	
	void fromRotationMatrix(const float matrix[3][3], double& w, double& x, double& y, double& z);
	double calculateAngleWithGroundPlane(double A, double B, double C);
	double magnitude(const double a, const double b, const double c);
	double dotProduct(const double a1, const double a2, const double b1, const double b2);

	void move_dist(std::vector<float> &cloud, std::vector<double>& move);
};