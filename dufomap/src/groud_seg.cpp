#include "groud_seg.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <iostream>

#include <random>
//#include "dipgseg.h"
#include "patchworkpp.h"
#include "groud_seg.h"
#include"AABB.h"
#include "ransac.h"

using namespace std;
//using namespace Potree;

Groud_seg::Groud_seg()
{
}

Groud_seg::~Groud_seg()
{
}

// ����ת����ת��Ϊ��Ԫ��
void Groud_seg::fromRotationMatrix(const float matrix[3][3], double& w, double& x, double& y, double& z) {
	float trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
	float s;

	if (trace > 0) {
		s = 0.5f / sqrt(trace + 1.0f);
		w = 0.25f / s;
		x = (matrix[2][1] - matrix[1][2]) * s;
		y = (matrix[0][2] - matrix[2][0]) * s;
		z = (matrix[1][0] - matrix[0][1]) * s;
	}
	else {
		if (matrix[0][0] > matrix[1][1] && matrix[0][0] > matrix[2][2]) {
			s = 2.0f * sqrt(1.0f + matrix[0][0] - matrix[1][1] - matrix[2][2]);
			w = (matrix[2][1] - matrix[1][2]) / s;
			x = 0.25f * s;
			y = (matrix[0][1] + matrix[1][0]) / s;
			z = (matrix[0][2] + matrix[2][0]) / s;
		}
		else if (matrix[1][1] > matrix[2][2]) {
			s = 2.0f * sqrt(1.0f + matrix[1][1] - matrix[0][0] - matrix[2][2]);
			w = (matrix[0][2] - matrix[2][0]) / s;
			x = (matrix[0][1] + matrix[1][0]) / s;
			y = 0.25f * s;
			z = (matrix[1][2] + matrix[2][1]) / s;
		}
		else {
			s = 2.0f * sqrt(1.0f + matrix[2][2] - matrix[0][0] - matrix[1][1]);
			w = (matrix[1][0] - matrix[0][1]) / s;
			x = (matrix[0][2] + matrix[2][0]) / s;
			y = (matrix[1][2] + matrix[2][1]) / s;
			z = 0.25f * s;
		}
	}
}

bool Groud_seg::groud_seg_writePLY_ascii(const std::string& filename,
	std::vector<float>  &cloud_ground,
	vector<unsigned char>&ground_color,
	std::vector<float>  &non_cloud_ground,
	vector<unsigned char>&non_ground_color)
{
	long long pointsProcessed = 0;

	long point_num = cloud_ground.size()/3 + non_cloud_ground.size()/3;
	std::ofstream writeout(filename.c_str(), std::ios::trunc);
	if (!writeout.good()) {
		std::cout << "create ply failed. file: " << filename.c_str() << std::endl;
		return false;
	}

	writeout << "ply" << "\n";
	writeout << "format ascii 1.0" << "\n";
	writeout << "element vertex " << point_num << "\n";
	writeout << "property float x" << "\n";
	writeout << "property float y" << "\n";
	writeout << "property float z" << "\n";

	writeout << "property uchar red" << "\n";
	writeout << "property uchar green" << "\n";
	writeout << "property uchar blue" << "\n";
	writeout << "end_header" << "\n";

	for (int i = 0; i < cloud_ground.size()/3; i++) {
		unsigned char r = ground_color[0];
		unsigned char g = ground_color[1];
		unsigned char b = ground_color[2];

		float x = cloud_ground[3*i+0];
		float y = cloud_ground[3*i+1];
		float z = cloud_ground[3*i+2];

		Potree::Vector3<float> p(x, y, z);
		Potree::Vector3<unsigned char> c(r, g, b);

		// writeout << p.x << " " << p.y << " " << p.z << " " << n.x << " " << n.y << " " << n.z << " " << (int)c.x << " " << (int)c.y << " " << (int)c.z << endl;
		writeout << p.x << " " << p.y << " " << p.z << " " << (int)c.x << " " << (int)c.y << " " << (int)c.z << std::endl;
	}

	for (int i = 0; i < non_cloud_ground.size()/3; i++) {
		unsigned char r = non_ground_color[0];
		unsigned char g = non_ground_color[1];
		unsigned char b = non_ground_color[2];

		float x = cloud_ground[3*i+0];
		float y = cloud_ground[3*i+1];
		float z = cloud_ground[3*i+2];

		Potree::Vector3<float> p(x, y, z);
		Potree::Vector3<unsigned char> c(r, g, b);

		// writeout << p.x << " " << p.y << " " << p.z << " " << n.x << " " << n.y << " " << n.z << " " << (int)c.x << " " << (int)c.y << " " << (int)c.z << endl;
		writeout << p.x << " " << p.y << " " << p.z << " " << (int)c.x << " " << (int)c.y << " " << (int)c.z << std::endl;
	}

	writeout.close();

	return true;
}

void Groud_seg::move_dist(std::vector<float>  & cloud, std::vector<double>&move) {
	double max_z = 0.0;
	double min_z = -2.0;
	double r = 2.0;

	int count = 0;
	int count_x_f = 0; // x ǰ
	int count_x_b = 0; // x ��

	for (int i = 0; i < cloud.size()/3; i++) {
		float x = cloud[3*i + 0];
		float y = cloud[3*i + 1];
		float z = cloud[3*i + 2];

		if (z > min_z && z < max_z && x * x + y * y < r * r) 
		{
			count++;
			if (abs(y) < 0.07) { // x ��
				if(x > 0)count_x_f++;
				if (x < 0)count_x_b++;
			}
		}
	}

	if (count > 500) {

		double move_dist_tmp = 10;
		if (count_x_f < count_x_b) {
			move[0] = -move_dist_tmp;
			move[1] = 0;
		}
		else {
			move[0] = move_dist_tmp;
			move[1] = 0;
		}
				
	}
}

void Groud_seg::groud_seg_patchworkpp(std::vector<float>  &cloud, std::string out_path) {

	vector<double> move_xy(2);
	move_xy[0] = 0;
	move_xy[1] = 0;
	move_dist(cloud, move_xy);

	patchwork::Params patchwork_parameters;
	patchwork_parameters.verbose = true;

	patchwork::PatchWorkpp Patchworkpp(patchwork_parameters);

	// Load point cloud
	Eigen::MatrixXf cloud_eig;
	cloud_eig.resize(cloud.size()/3, 4);
	
	for (int i = 0; i < cloud.size()/3; i++)
	{
		float x = cloud[3*i + 0] - move_xy[0];
		float y = cloud[3*i + 1] - move_xy[1];
		float z = cloud[3*i + 2];
		
		float intensity = 1.0;
		cloud_eig.row(i) << x, y, z, intensity;
	}

	// Estimate Ground
	Patchworkpp.estimateGround(cloud_eig);

	// Get Ground and Nonground
	//Eigen::MatrixX3f ground = Patchworkpp.getGround();
	//Eigen::MatrixX3f nonground = Patchworkpp.getNonground();
	//double time_taken = Patchworkpp.getTimeTaken();

	Eigen::VectorXi ground_idx = Patchworkpp.getGroundIndices();
	Eigen::VectorXi nonground_idx = Patchworkpp.getNongroundIndices();

	vector<int> vec_label(cloud.size()/3);
	for (int i = 0; i < ground_idx.rows(); i++) {
		int x = ground_idx.row(i)[0];

		if (cloud[3*x + 2] < 0.5) {
			vec_label[x] = 1; // �����Ϊ 1
		}
	}

	std::string filename = out_path;
	std::ofstream writeout(filename.c_str(), std::ios::trunc);
	if (!writeout.good()) {
		std::cout << "create ply failed. file: " << filename.c_str() << std::endl;
	}

	writeout << "ply" << "\n";
	writeout << "format ascii 1.0" << "\n";
	writeout << "element vertex " << vec_label.size() << "\n";
	writeout << "property float x" << "\n";
	writeout << "property float y" << "\n";
	writeout << "property float z" << "\n";

	writeout << "property uchar red" << "\n";
	writeout << "property uchar green" << "\n";
	writeout << "property uchar blue" << "\n";
	writeout << "end_header" << "\n";

	for (int i = 0; i < vec_label.size(); i++)
	{
		float x = cloud[3*i + 0];
		float y = cloud[3*i + 1];
		float z = cloud[3*i + 2];

		Potree::Vector3<unsigned char> c(255, 0, 0);

		if (vec_label[i] == 1) { // groud
			writeout << x << " " << y << " " << z << " " << (int)c.x << " " << (int)c.y << " " << (int)c.z << "\n";

		}
		else {
			c.x = 0;
			c.y = 255;
			c.z = 0;

			writeout << x << " " << y << " " << z << " " << (int)c.x << " " << (int)c.y << " " << (int)c.z << "\n";

		}
	}
	writeout.close();

	int ggg = 3;
}

// �������������ĵ��
double Groud_seg::dotProduct(const double a1, const double a2, const double b1, const double b2) {
	return a1 * b1 + a2 * b2;
}

// ����������ģ��
double Groud_seg::magnitude(const double a, const double b, const double c) {
	return sqrt(a * a + b * b + c * c);
}

// ����ƽ�����ƽ��ļн�
double Groud_seg::calculateAngleWithGroundPlane(double A, double B, double C) {
	// ��ƽ��ķ�����
	double x = 0, y = -1, z = 0;
	// ƽ��ķ�����
	double planeNormalX = A, planeNormalY = B, planeNormalZ = C;

	// ������
	double dot = dotProduct(planeNormalX, planeNormalY, x, y) + planeNormalZ * z;
	// ����ƽ�淨������ģ��
	double planeNormalMag = magnitude(A, B, C);
	// �����ƽ�淨������ģ����������1����Ϊ��ƽ��ķ�������(0, 0, 1)��
	double groundNormalMag = 1.0;

	// ����нǵ�����ֵ
	double cosTheta = dot / (planeNormalMag * groundNormalMag);
	// �������ڸ������������⵼��cosTheta��΢����-1��1�ķ�Χ
	cosTheta = std::max(-1.0, std::min(1.0, cosTheta));

	// ����н�
	return acos(cosTheta);
}

void Groud_seg::groud_seg_my(std::vector<float>  &cloud, 
std::vector<float>  &out_ground_cloud,
std::vector<float>  &out_non_ground_cloud,
 std::string out_ply_path) {

	 if(cloud.size() / 3 < 50){
		 out_non_ground_cloud = cloud;
		 return;
	 }

	cv::Mat point_cloud;
	point_cloud = cv::Mat(cloud.size() / 3, 3, CV_32F, cloud.data());
	//memcpy(point_cloud.data, xyz_vec.data(), xyz_vec.size() * sizeof(float));

	cv::Mat labels;
	std::vector<cv::Vec4f> planes;

	// ƽ����
	int desired_num_planes = 1;
	float thr = 0.07;
	float grid_size = 0.0;
	int max_iters = 1000;
	//float nor1 = 0, nor2 = 0, nor3 = 1;
	float nor1 = 0, nor2 = -1, nor3 = 0;

	cv::Vec3f normal(nor1, nor2, nor3);
	cv::Vec3f* normal_ptr;
	if (nor1 == 0 && nor2 == 0 && nor3 == 0) {
		normal_ptr = nullptr;
	}
	else {
		normal_ptr = &normal;
	}
	auto start_all = std::chrono::high_resolution_clock::now();
	get_planes(labels, planes, point_cloud, thr, max_iters, desired_num_planes, grid_size, normal_ptr);
	long long duration_block = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_all).count();
	//g_logf("\n\get_planes total cost time: %.3f s \n", duration_block / 1000.0f);

	if(planes.size() != 1){
		out_non_ground_cloud = cloud;
		return;
	}
	
	std::vector<int>vec_label(cloud.size() / 3);
	long long count = 0;

	std::vector<float>  out_ground_cloud_tmp;
	std::vector<float>  out_non_ground_cloud_tmp;
	if (planes.size() == 1) {
		double a = planes[0][0];
		double b = planes[0][1];
		double c = planes[0][2];
		double d = planes[0][3];

		double n = sqrt(a * a + b * b + c * c);
		long under_groud_pts_num = 0;
		long under_groud_pts_num2 = 0;
		for (int i = 0; i < cloud.size() / 3; i++) {
			double x = cloud[3 * i];
			double y = cloud[3 * i + 1];
			double z = cloud[3 * i + 2];

			double dis = abs((a * x + b * y + c * z + d)) / n;

			// ƽ���²�
			if (a * x + b * y + c * z + d < 0 && dis > thr * 2) {
				under_groud_pts_num++;
			}
			if (a * x + b * y + c * z + d > 0 && dis > thr * 2) {
				under_groud_pts_num2++;
			}

			if (dis > thr)
			{
				vec_label[i] = 1;
				count++;
				out_non_ground_cloud_tmp.push_back(x);
				out_non_ground_cloud_tmp.push_back(y);
				out_non_ground_cloud_tmp.push_back(z);
			}
			else {
				vec_label[i] = 0;
				out_ground_cloud_tmp.push_back(x);
				out_ground_cloud_tmp.push_back(y);
				out_ground_cloud_tmp.push_back(z);
			}
		}

		out_ground_cloud = out_ground_cloud_tmp;
		out_non_ground_cloud = out_non_ground_cloud_tmp;
		//ƽ�����ƽ��н�
		double jiaodu = abs(calculateAngleWithGroundPlane(a, b, c) / PI * 180.0);
		double rate = (double)min(under_groud_pts_num, under_groud_pts_num2) / vec_label.size();
		int gg34 = vec_label.size();
		// if (rate > 0.05) {
		// 	cout << "\n\groud too high. \n" << endl;

		// 	for (int i = 0; i < cloud.size() / 3; i++) {
		// 		vec_label[i] = 1;
		// 	}
		// }

		if(out_ply_path != ""){
			std::string filename = out_ply_path;
			std::ofstream writeout(filename.c_str(), std::ios::trunc);
			if (!writeout.good()) {
				std::cout << "create ply failed. file: " << filename.c_str() << std::endl;

			}

			writeout << "ply" << "\n";
			writeout << "format ascii 1.0" << "\n";
			writeout << "element vertex " << cloud.size() / 3 << "\n";
			writeout << "property float x" << "\n";
			writeout << "property float y" << "\n";
			writeout << "property float z" << "\n";

			writeout << "property uchar red" << "\n";
			writeout << "property uchar green" << "\n";
			writeout << "property uchar blue" << "\n";
			writeout << "end_header" << "\n";

			for (int i = 0; i < cloud.size() / 3; i++)
			{
				float x = cloud[3 * i];
				float y = cloud[3 * i + 1];
				float z = cloud[3 * i + 2];

				Potree::Vector3<unsigned char> c(0, 255, 0);

				if (vec_label[i] == 1) {
					writeout << x << " " << y << " " << z << " " << (int)c.x << " " << (int)c.y << " " << (int)c.z << "\n";

				}
				else { // groud
					c.x = (unsigned char)255;
					c.y = (unsigned char)0;
					c.z =(unsigned char) 0;

					writeout << x << " " << y << " " << z << " " << (int)c.x << " " << (int)c.y << " " << (int)c.z << "\n";

				}
			}
			writeout.close();

		}
		
		int gg = 0;
	}
}
