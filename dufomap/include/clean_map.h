#pragma once

//Open3D
//0.9.0版Open3D的绝对路径头文件
#include </usr/local/include/Open3D/Open3D.h>

#include"clean_map_def.h"
#include <opencv2/opencv.hpp>
#include <opencv2/flann/flann.hpp>

class CleanMap {
private:
    std::string m_data_root;
    std::string m_config_path;
    Config m_config;
    std::map<std::string, SweepLocation> m_map_id_sweepLocation;

    std::string m_las_dir;
	std::string m_ply_dir ;
	std::string m_ground_dir;
	std::string m_new_laz_dir;

    //open3d
    void filterGroundPlane_2_open3d(const std::shared_ptr<open3d::geometry::PointCloud> &pc,
    std::shared_ptr<open3d::geometry::PointCloud> &ground,
    std::shared_ptr<open3d::geometry::PointCloud> &nonground,
    double groundFilterDistance,
    double groundFilterAngle);

    void retainMapRange(std::map<std::string, SweepLocation>& map, size_t n, size_t m);

    void filterNoise_open3d(ufo::PointCloudColor &cloud, ufo::PointCloudColor &out_cloud_noise, Potree::AABB &out_aabb);
    void get_camera_traj(ufo::PointCloudColor &cloud_camera_traj, cv::Mat &out_pointsA);

    template <class Map>
    void cluster(Map& map, Clustering const& clustering)
    {
        std::unordered_set<ufo::Node> seen;
        std::vector<ufo::Sphere>      queue;

        auto depth        = clustering.depth;
        auto max_distance = clustering.max_distance;
        auto min_points   = clustering.min_points;

        ufo::label_t l{1};
        for (auto node : map.query(ufo::pred::Leaf(depth) && ufo::pred::SeenFree() &&
                                ufo::pred::HitsMin(1) && ufo::pred::Label(0))) {
            if (map.label(node.index())) {  // FIXME: This is because how the iterator works
                continue;
            }

            seen = {node};
            queue.assign(1, ufo::Sphere(map.center(node), max_distance));

            map.setLabel(node, l, false);

            while (!queue.empty()) {
                auto p = ufo::pred::Intersects(queue);
                queue.clear();
                for (auto const& node : map.query(
                        ufo::pred::Leaf(depth) && ufo::pred::SeenFree() && ufo::pred::HitsMin(1) &&
                        ufo::pred::Label(0) && std::move(p) &&
                        ufo::pred::Satisfies([&seen](auto n) { return seen.insert(n).second; }))) {
                    queue.emplace_back(map.center(node), max_distance);
                    map.setLabel(node, l, false);
                }
            }

            if (seen.size() < min_points) {
                for (auto e : seen) {
                    if (l == map.label(e)) {
                        map.setLabel(e, -1, false);
                    }
                }
            }

            l += seen.size() >= min_points;

            map.propagateModified();  // FIXME: Should this be here?
        }
    }

    Config readConfig(std::filesystem::path path);

public:
    CleanMap(std::string data_root, std::string config_path);

    ~CleanMap();

    void init();

    void get_id_sweepLocation(std::map<std::string, SweepLocation>&map_id_sweepLocation);

    void run();
};
