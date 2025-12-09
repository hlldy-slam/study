// TOML
#include "toml.hpp"
#include "indicators.hpp"

// UFO
#include <ufo/map/integration/integration.hpp>
#include <ufo/map/integration/integration_parameters.hpp>
#include <ufo/map/node.hpp>
#include <ufo/map/point.hpp>
#include <ufo/map/point_cloud.hpp>
#include <ufo/map/points/points_predicate.hpp>
#include <ufo/map/predicate/satisfies.hpp>
#include <ufo/map/predicate/spatial.hpp>
#include <ufo/map/types.hpp>
#include <ufo/map/ufomap.hpp>
#include <ufo/math/pose6.hpp>
#include <ufo/util/timing.hpp>

// STL
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef UFO_PARALLEL
// STL
#include <execution>
#endif

#include "json.hpp"
#include "groud_seg.h"
#include <Eigen/Dense>

#include "clean_map.h"
#include <omp.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <deque>
#include <condition_variable>

// protect concurrent std::cout usage from multiple threads
static std::mutex cout_mutex;

// ================= 预读取队列与条目 =================
struct CloudItem {
    SweepLocation sweep{};          // 位姿与路径
    ufo::PointCloudColor cloud;     // 读取的原始点云（局部坐标）
    Potree::AABB aabb;              // 包围盒
    ufo::Color color{};             // 随机或输入着色
    bool ok{false};                 // 读取是否成功
};

class CloudQueue {
public:
    explicit CloudQueue(size_t cap) : capacity_(cap) {}
    void push(CloudItem item) {
        std::unique_lock<std::mutex> lk(m_);
        cv_full_.wait(lk, [&]{ return finished_ || queue_.size() < capacity_; });
        if (finished_) return; // 若已结束则丢弃
        queue_.push_back(std::move(item));
        cv_empty_.notify_one();
    }
    bool pop(CloudItem &out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_empty_.wait(lk, [&]{ return !queue_.empty() || finished_; });
        if (queue_.empty()) return false; // finished_ 且空
        out = std::move(queue_.front());
        queue_.pop_front();
        cv_full_.notify_one();
        return true;
    }
    void finish() {
        std::lock_guard<std::mutex> lk(m_);
        finished_ = true;
        cv_empty_.notify_all();
        cv_full_.notify_all();
    }
private:
    std::deque<CloudItem> queue_;
    size_t capacity_;
    bool finished_{false};
    std::mutex m_;
    std::condition_variable cv_empty_, cv_full_;
};

// 读取单个 sweep 点云
static CloudItem readCloudItem(const SweepLocation &sweep, std::mt19937 &gen) {
    CloudItem ci;
    ci.sweep = sweep;
    std::uniform_int_distribution<> dis(0,255);
    ci.color.red = dis(gen); ci.color.green = dis(gen); ci.color.blue = dis(gen);
    try {
        ufo::readPointCloudLAZ(sweep.model_path, ci.cloud, ci.aabb, ci.color);
        ci.ok = true;
    } catch (...) {
        ci.ok = false;
    }
    return ci;
}

CleanMap::CleanMap(std::string data_root, std::string config_path)
 {
    m_data_root = data_root; 
    m_config_path = config_path;

    m_las_dir = m_data_root + "capture/";
	m_ply_dir = m_data_root + "ply/";
	m_ground_dir =  m_data_root + "ground/";
	m_new_laz_dir = m_data_root + "new_laz/";
}

CleanMap::~CleanMap()
{

}

Config CleanMap::readConfig(std::filesystem::path path)
{
	Config config;
	for (;;) {
		if (std::filesystem::exists(path)) {
			toml::table tbl;
			try {
				tbl = toml::parse_file((path).string());
			} catch (toml::parse_error const& err) {
				std::cerr << "Configuration parsing failed:\n" << err << '\n';
				exit(1);
			}

			config.read(tbl);
			if (config.printing.verbose) {
				std::cout << "Found: " << (path) << '\n';
			}

			break;
		}
		if (!path.has_parent_path()) {
			std::cout << "Did not find configuration file, using default.\n";
			break;
		}
		path = path.parent_path();
	}

	if (config.printing.verbose) {
		std::cout << config << '\n';
	}

	return config;
}

void CleanMap::init(){
	std::cout << "init begin" << std::endl;

    m_config = readConfig(std::filesystem::path(m_config_path));
    get_id_sweepLocation(m_map_id_sweepLocation);

   // retainMapRange 从1开始
	size_t select_m = 0;
	size_t select_n = m_map_id_sweepLocation.size() - 1;

	if(m_config.fdageParam.select_m >= 0 && m_config.fdageParam.select_n >= 0 
	&& m_config.fdageParam.select_m <= m_config.fdageParam.select_n){
		select_m = m_config.fdageParam.select_m;
		select_n = m_config.fdageParam.select_n;
	}
	retainMapRange(m_map_id_sweepLocation, select_m+1, select_n+1);
	std::cout << "init end" << std::endl;
}

static std::pair<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>>
RemoveStatisticalOutlierOpen3D(const std::shared_ptr<open3d::geometry::PointCloud> &pc,
                               int nb_neighbors, double std_ratio)
{
    if (!pc || pc->points_.empty()) {
        return {std::make_shared<open3d::geometry::PointCloud>(), {}};
    }

    // build KDTree
    open3d::geometry::KDTreeFlann kdtree(*pc);

    size_t n = pc->points_.size();
    std::vector<double> mean_dists(n, std::numeric_limits<double>::infinity());

    std::vector<int> indices_tmp;
    std::vector<double> dists_tmp;
    for (size_t i = 0; i < n; ++i) {
        // search nb_neighbors+1 to skip the point itself if returned
        int found = kdtree.SearchKNN(pc->points_[i], nb_neighbors + 1, indices_tmp, dists_tmp);
        if (found <= 1) {
            mean_dists[i] = std::numeric_limits<double>::infinity();
            continue;
        }
        // dists_tmp are squared distances in many Open3D versions -> take sqrt
        double sum = 0.0;
        int count = 0;
        for (int k = 0; k < found; ++k) {
            if (indices_tmp[k] == (int)i) continue; // skip self
            sum += std::sqrt(dists_tmp[k]);
            ++count;
        }
        if (count > 0) mean_dists[i] = sum / static_cast<double>(count);
        else mean_dists[i] = std::numeric_limits<double>::infinity();
    }

    // compute global mean and stddev of mean_dists for valid entries
    double sum = 0.0;
    double sumsq = 0.0;
    size_t valid = 0;
    for (double v : mean_dists) {
        if (std::isfinite(v)) {
            sum += v;
            sumsq += v * v;
            ++valid;
        }
    }
    if (valid == 0) {
        return {std::make_shared<open3d::geometry::PointCloud>(), {}};
    }
    double mean = sum / valid;
    double var = sumsq / valid - mean * mean;
    double stddev = var > 0.0 ? std::sqrt(var) : 0.0;
    double threshold = mean + std_ratio * stddev;

    std::vector<size_t> inlier_indices;
    inlier_indices.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (mean_dists[i] <= threshold) inlier_indices.push_back(i);
    }

    auto inlier_pc = std::make_shared<open3d::geometry::PointCloud>();
    inlier_pc->points_.reserve(inlier_indices.size());
    if (!pc->colors_.empty()) inlier_pc->colors_.reserve(inlier_indices.size());
    if (!pc->normals_.empty()) inlier_pc->normals_.reserve(inlier_indices.size());

    for (size_t idx : inlier_indices) {
        inlier_pc->points_.push_back(pc->points_[idx]);
        if (!pc->colors_.empty()) inlier_pc->colors_.push_back(pc->colors_[idx]);
        if (!pc->normals_.empty()) inlier_pc->normals_.push_back(pc->normals_[idx]);
    }

    return {inlier_pc, inlier_indices};
}


void CleanMap::get_id_sweepLocation(std::map<std::string, SweepLocation>&out_map_id_sweepLocation){
    std::string vision_json_path = m_data_root + "capture/slam_data_new.json";

    // ifstream ifs(vision_json_path, ios::in);
	// auto j = nlohmann::json::parse(ifs);
	// ifs.close();
	/////////////////兼容麒麟v10 aarch64
nlohmann::json j;
try {
	ifstream ifs(vision_json_path, ios::in);
	j = nlohmann::json::parse(ifs);
	ifs.close();
}
catch (exception e) {
	j.clear();
	ifstream ifs(vision_json_path, ios::in);
	if (char ch = ifs.get(); ch == ' ')
	{
		return;
	}
	while (ifs.good())
	{
		ifs.unget();
		ifs >> j;
		ifs.get();
		break;
	}
	ifs.close();
}
/////////////////////////

    std::map<std::string, SweepLocation>map_id_sweepLocation;

	for (int floor_index = 0; floor_index < j["views_info"].size(); floor_index++) {
	    auto& j_floor_ele = j["views_info"][floor_index];

	    int floor_id = j_floor_ele["id_floor"];
        int location_num = j_floor_ele["list_pose"].size();

		for(int i = 0; i < location_num; i++){
			auto j_pose =j_floor_ele["list_pose"][i];

			std::string uuid = std::to_string(i);

			//insert map  xx_xx  make key
			int id0 = floor_id;
			int id = i;

			std::ostringstream tmp_id0;
			tmp_id0  << std::setfill('0') << std::setw(6) << id0;
			std::ostringstream tmp_id;
			tmp_id  << std::setfill('0') << std::setw(6) << id;

			std::string str_id = tmp_id0.str() + tmp_id.str();

			SweepLocation ele;
			ele.uuid = uuid;
			ele.floor_id = floor_id;
			ele.model_path = m_las_dir + std::to_string(floor_id) + "_" + uuid + "_depth.laz";
			
			ele.pose6f.x() = j_pose[4];
			ele.pose6f.y() = j_pose[5];
			ele.pose6f.z() = j_pose[6];

			ele.pose6f.qw() = j_pose[0];
			ele.pose6f.qx() = j_pose[1];
			ele.pose6f.qy() = j_pose[2];
			ele.pose6f.qz() = j_pose[3];

			if(std::filesystem::exists(ele.model_path)){
				map_id_sweepLocation.insert(std::make_pair(str_id, ele));
			}
		
    	}
	}

    out_map_id_sweepLocation = map_id_sweepLocation;
}

void CleanMap::retainMapRange(std::map<std::string, SweepLocation>& map, size_t n, size_t m) {
        if (n > m || m > map.size()) return; // 确保范围有效
    
        auto begin_it = std::next(map.begin(), n - 1);
        auto end_it = std::next(map.begin(), m);
    
        map.erase(map.begin(), begin_it);
        map.erase(end_it, map.end());
}

void CleanMap::filterGroundPlane_2_open3d(const std::shared_ptr<open3d::geometry::PointCloud> &pc,
    std::shared_ptr<open3d::geometry::PointCloud> &ground,
    std::shared_ptr<open3d::geometry::PointCloud> &nonground,
    double groundFilterDistance,
    double groundFilterAngle) {
    if (!pc || pc->points_.size() < 50) {
        std::cout << "Pointcloud too small, skipping ground plane extraction\n";
        nonground = pc;
        return;
    }

    // 估计法线（如果没有）
    if (pc->normals_.empty()) {
        pc->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    }

    // RANSAC 平面分割
    int ransac_n = 3;
    int num_iterations = 1000;
    double distance_threshold = groundFilterDistance;
    Eigen::Vector4d plane_model;
    std::vector<size_t> inliers;
    std::tie(plane_model, inliers) = pc->SegmentPlane(distance_threshold, ransac_n, num_iterations);

    if (inliers.empty()) {
        std::cout << "Open3D segmentation did not find any plane.\n";
        nonground = pc;
        return;
    }

    // 判断分割出的平面法线是否和期望地面轴接近（示例中期望轴为 Y 方向 (0,1,0)）
    Eigen::Vector3d plane_normal = plane_model.head<3>();
    plane_normal.normalize();
    Eigen::Vector3d axis(0.0, 1.0, 0.0); // 原 PCL 代码使用 (0,1,0)
    double angle = std::acos(std::abs(plane_normal.dot(axis))); // 0..pi/2
    if (angle > groundFilterAngle) {
        // 不是期望方向的平面 -> 视为未找到地面
        std::cout << "Found plane but angle too large, skipping as ground.\n";
        nonground = pc;
        return;
    }

    // SelectDownSample 在不同版本可能返回 PointCloud（按值）或 shared_ptr<PointCloud>，兼容两种情况
    auto tmp_ground = pc->SelectDownSample(inliers);
    using TmpGroundT = std::decay_t<decltype(tmp_ground)>;
    if constexpr (std::is_same_v<TmpGroundT, std::shared_ptr<open3d::geometry::PointCloud>>) {
        ground = tmp_ground;
    } else {
            ground = std::make_shared<open3d::geometry::PointCloud>(std::move(tmp_ground));
    }

    // nonground：构造索引补集，然后用 SelectDownSample
    std::vector<char> is_inlier(pc->points_.size(), 0);
    for (auto idx : inliers) {
        if (idx < is_inlier.size()) is_inlier[idx] = 1;
    }
    std::vector<size_t> complement;
    complement.reserve(pc->points_.size() - inliers.size());
    for (size_t i = 0; i < is_inlier.size(); ++i) {
        if (!is_inlier[i]) complement.push_back(i);
    }

    auto tmp_nonground = pc->SelectDownSample(complement);
    using TmpNonGroundT = std::decay_t<decltype(tmp_nonground)>;
    if constexpr (std::is_same_v<TmpNonGroundT, std::shared_ptr<open3d::geometry::PointCloud>>) {
        nonground = tmp_nonground;
    } else {
        nonground = std::make_shared<open3d::geometry::PointCloud>(std::move(tmp_nonground));
    }	
}

void CleanMap::filterNoise_open3d(ufo::PointCloudColor &cloud, ufo::PointCloudColor &out_cloud_noise, Potree::AABB &out_aabb)
{
    // 构造 open3d 点云（跳过 NaN/inf）
    auto o3d_pc = std::make_shared<open3d::geometry::PointCloud>();
    o3d_pc->points_.reserve(cloud.size());
    bool has_color = true;
    for (size_t i = 0; i < cloud.size(); ++i) {
        double x = cloud[i].x;
        double y = cloud[i].y;
        double z = cloud[i].z;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
        o3d_pc->points_.push_back(Eigen::Vector3d(x, y, z));
        // color in ufo is 0..255
        o3d_pc->colors_.push_back(Eigen::Vector3d(cloud[i].red/255.0, cloud[i].green/255.0, cloud[i].blue/255.0));
    }

    // 如果点云为空，直接返回空结果
    if (o3d_pc->points_.empty()) {
        out_aabb = Potree::AABB();
        cloud.clear();
        out_cloud_noise.clear();
        return;
    }

    bool filterNoise = m_config.filterNoise.enable;
    std::shared_ptr<open3d::geometry::PointCloud> inlier_pc;
    std::vector<size_t> inlier_indices;

    if (filterNoise) {
        int mean_k = m_config.filterNoise.filterMeanK;
        double std_ratio = m_config.filterNoise.StddevMulThresh;
        std::tie(inlier_pc, inlier_indices) = RemoveStatisticalOutlierOpen3D(o3d_pc, mean_k, std_ratio);
    } else {
        // 不做统计去噪，只去除非有限点（已经在构建时跳过）
        inlier_pc = o3d_pc;
        inlier_indices.resize(inlier_pc->points_.size());
        for (size_t i = 0; i < inlier_indices.size(); ++i) inlier_indices[i] = i;
    }

    // 生成噪声点 (补集)
    std::vector<char> is_inlier(o3d_pc->points_.size(), 0);
    for (auto idx : inlier_indices) if (idx < is_inlier.size()) is_inlier[idx] = 1;
    std::vector<size_t> noise_indices;
    noise_indices.reserve(o3d_pc->points_.size() - inlier_indices.size());
    for (size_t i = 0; i < is_inlier.size(); ++i) if (!is_inlier[i]) noise_indices.push_back(i);

    // 把 inlier_pc 转回 ufo::PointCloudColor 并更新 out_aabb
    ufo::PointCloudColor out_cloud(inlier_pc->points_.size());
    for (size_t i = 0; i < inlier_pc->points_.size(); ++i) {
        const auto &p = inlier_pc->points_[i];
        out_cloud[i].x = p.x();
        out_cloud[i].y = p.y();
        out_cloud[i].z = p.z();
        if (!inlier_pc->colors_.empty()) {
            out_cloud[i].red   = static_cast<unsigned char>(std::clamp(int(std::round(inlier_pc->colors_[i].x()*255.0)), 0, 255));
            out_cloud[i].green = static_cast<unsigned char>(std::clamp(int(std::round(inlier_pc->colors_[i].y()*255.0)), 0, 255));
            out_cloud[i].blue  = static_cast<unsigned char>(std::clamp(int(std::round(inlier_pc->colors_[i].z()*255.0)), 0, 255));
        } else {
            out_cloud[i].red = out_cloud[i].green = out_cloud[i].blue = 255;
        }
        Potree::Vector3<double> pt_tmp(out_cloud[i].x, out_cloud[i].y, out_cloud[i].z);
        out_aabb.update(pt_tmp);
    }
    cloud = out_cloud;

    // 构造噪声点云返回
    ufo::PointCloudColor out_noise(noise_indices.size());
    for (size_t k = 0; k < noise_indices.size(); ++k) {
        size_t idx = noise_indices[k];
        const auto &p = o3d_pc->points_[idx];
        out_noise[k].x = p.x();
        out_noise[k].y = p.y();
        out_noise[k].z = p.z();
        if (!o3d_pc->colors_.empty()) {
            out_noise[k].red   = static_cast<unsigned char>(std::clamp(int(std::round(o3d_pc->colors_[idx].x()*255.0)), 0, 255));
            out_noise[k].green = static_cast<unsigned char>(std::clamp(int(std::round(o3d_pc->colors_[idx].y()*255.0)), 0, 255));
            out_noise[k].blue  = static_cast<unsigned char>(std::clamp(int(std::round(o3d_pc->colors_[idx].z()*255.0)), 0, 255));
        } else {
            out_noise[k].red = out_noise[k].green = out_noise[k].blue = 0;
        }
    }
    out_cloud_noise = out_noise;
}

void CleanMap::get_camera_traj(ufo::PointCloudColor &cloud_camera_traj, cv::Mat &out_pointsA)
{
		int loc_num = m_map_id_sweepLocation.size();
		ufo::PointCloudColor cloud_camera_traj_tmp(loc_num);

		int i = 0;
		// cv::Mat pointsA(loc_num, 3, CV_32F);

		for(auto& pair:m_map_id_sweepLocation){
		SweepLocation ele = pair.second;
		std::string las_path = ele.model_path;
		ufo::Pose6f viewpoint = ele.pose6f;

		double x = viewpoint.translation.x;
		double y = viewpoint.translation.y;
		double z = viewpoint.translation.z;

		cloud_camera_traj_tmp[i].x = x; 
		cloud_camera_traj_tmp[i].y = y;
		cloud_camera_traj_tmp[i].z = z;

		cloud_camera_traj_tmp[i].red   = 255;
		cloud_camera_traj_tmp[i].green   =0;
		cloud_camera_traj_tmp[i].blue   = 255;

		out_pointsA.at<float>(i, 0) = x;
		 out_pointsA.at<float>(i, 1) = y; 
		 out_pointsA.at<float>(i, 2) = z;

		i++;
	}

	cloud_camera_traj = cloud_camera_traj_tmp;
}

void CleanMap::run_prefetch() {
    auto begin_t = high_resolution_clock::now();

    // 全局线程统计容器（函数内静态，避免重复定义）
    struct ThreadStats {
        double io_ms=0;                 // 读取 + 初步复制
        double insert_ms=0;             // 总插入时间 (锁持有 + 调用)
        double insert_lock_wait_ms=0;   // 等待锁时间估计
        double insert_call_ms=0;        // 真正调用 insertPointCloud 时间
        double transform_ms=0;          // applyTransform 耗时
        double build_points_ms=0;       // 构建 vec_points + N/TN 等矩阵准备
        double queue_wait_ms=0;         // pop 等待时间估计
        double ground_ms=0;             // 地面分割
        double downsample_ms=0;         // 体素下采样
        double classify_ms=0;           // 分类总时间
        double classify_flann_ms=0;     // FLANN 最近邻耗时
        double classify_seenfree_ms=0;  // seenFree 查询耗时
        size_t sweeps=0;                // 处理的 sweep 数
        size_t points_read=0;           // 原始读取点数
        size_t points_after_ds=0;       // 下采样后点数
    };
    std::vector<ThreadStats> thread_stats_all; thread_stats_all.reserve(32);

    // 摄像机轨迹
    cv::Mat pointsA(m_map_id_sweepLocation.size(), 3, CV_32F);
    ufo::PointCloudColor cloud_camera_traj;
    get_camera_traj(cloud_camera_traj, pointsA);

    ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION | ufo::MapType::LABEL> map(
        m_config.map.resolution, m_config.map.levels);
    map.reserve(100'000'000);

    Potree::AABB all_aabb;
    // 如果单消费者处理，可去掉 map_mutex；保留也没问题
    std::mutex map_mutex;

    // SweepLocation 向量
    std::vector<SweepLocation> vec_sweepLocation;
    vec_sweepLocation.reserve(m_map_id_sweepLocation.size());
    for (auto &kv : m_map_id_sweepLocation) vec_sweepLocation.push_back(kv.second);

    // 全局结果容器
    ufo::PointCloudColor cloud_static;
    ufo::PointCloudColor cloud_remove;
    ufo::PointCloudColor cloud_acc_ground;
    ufo::PointCloudColor cloud_acc_noise; // 如果需要噪声逻辑，可后续补

    // 队列 + 生产者线程
    size_t prefetch_depth = 16;
    CloudQueue queue(prefetch_depth);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::thread producer([&]{
        for (auto &sweep : vec_sweepLocation) {
            auto item = readCloudItem(sweep, gen);
            queue.push(std::move(item));
        }
        queue.finish();
    });

    // 多消费者实现
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 4; // 兜底
    unsigned worker_count = std::max(1u, hc - 1);
    if (worker_count > 8) worker_count = 8; // 简单上限，防止过度争用
    std::atomic<size_t> processed{0};
    std::mutex results_mutex; // 汇总锁

    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (unsigned wi = 0; wi < worker_count; ++wi) {
        workers.emplace_back([&, wi](){
            // 每线程本地缓冲，减少全局锁频度
            ufo::PointCloudColor local_static;
            ufo::PointCloudColor local_remove;
            ufo::PointCloudColor local_ground;
            // 每线程 FLANN 索引（避免潜在线程安全问题）
            cv::flann::Index localFlann(pointsA, cv::flann::KDTreeIndexParams(4));

            // 每线程性能统计
            ThreadStats stats;
            auto now_ms = [](){ return (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()/1000.0; };

            CloudItem item;
            while (true) {
                double t_pop_begin = now_ms();
                bool ok_pop = queue.pop(item);
                double t_pop_end = now_ms();
                if (!ok_pop) break;
                stats.queue_wait_ms += (t_pop_end - t_pop_begin);
                if (!item.ok) {
                    size_t cur = ++processed;
                    if (!m_config.printing.verbose) {
                        std::lock_guard<std::mutex> lk(cout_mutex);
                        std::cout << "(" << cur << "/" << vec_sweepLocation.size() << ") Skip invalid sweep\r" << std::flush;
                    }
                    continue;
                }

                // 保留原始局部坐标副本
                double t0 = now_ms();
                ufo::PointCloudColor cloud_raw = item.cloud;
                stats.points_read += cloud_raw.size();
                double t1_io_end = now_ms();
                stats.io_ms += (t1_io_end - t0); // I/O + 构造副本时间（读取已在 producer）

                // 世界坐标用于地图插入
                double t_transform_begin = now_ms();
                ufo::applyTransform(item.cloud, item.sweep.pose6f);
                stats.transform_ms += (now_ms() - t_transform_begin);
                // 插入前可选体素下采样（减少插入体量）: 针对世界坐标 item.cloud
                // 插入阶段细分: 估计锁等待 + 调用耗时
                double t_lock_req = now_ms();
                {
                    std::unique_lock<std::mutex> lk(map_mutex);
                    double t_lock_acq = now_ms();
                    stats.insert_lock_wait_ms += (t_lock_acq - t_lock_req);
                    double t_call_begin = t_lock_acq;
                    // 传入 propagate=false，统一在末尾一次 propagateModified
                    ufo::insertPointCloud(map, item.cloud, item.sweep.pose6f.translation,
                                          m_config.integration, /*propagate*/ false);
                    all_aabb.update(item.aabb);
                    double t_call_end = now_ms();
                    stats.insert_call_ms += (t_call_end - t_call_begin);
                    stats.insert_ms += (t_call_end - t_lock_req); // 总时间
                }

                // vec_points 构造（地面分割使用原始局部坐标）
                double t_build_begin = now_ms();
                std::vector<Potree::Point> vec_points(cloud_raw.size());
                for (size_t i = 0; i < cloud_raw.size(); ++i) {
                    Potree::Point p;
                    p.position.x = cloud_raw[i].x;
                    p.position.y = cloud_raw[i].y;
                    p.position.z = cloud_raw[i].z;
                    p.color.x = cloud_raw[i].red;
                    p.color.y = cloud_raw[i].green;
                    p.color.z = cloud_raw[i].blue;
                    vec_points[i] = p;
                }
                stats.build_points_ms += (now_ms() - t_build_begin);

                auto pc_ground = std::make_shared<open3d::geometry::PointCloud>();
                auto pc_nonground = std::make_shared<open3d::geometry::PointCloud>();
                bool bfilterGroundPlane = m_config.groundSeg.enable;
                if (bfilterGroundPlane) {
                    double t_ground_begin = now_ms();
                    auto cloud_cut_ground = std::make_shared<open3d::geometry::PointCloud>();
                    auto cloud_cut_nonground = std::make_shared<open3d::geometry::PointCloud>();
                    std::vector<float> vec_cloud_cut;
                    for (size_t ii = 0; ii < vec_points.size(); ++ii) {
                        Eigen::Vector3d p3(vec_points[ii].position.x, vec_points[ii].position.y, vec_points[ii].position.z);
                        Eigen::Vector3d n3(vec_points[ii].normal.x, vec_points[ii].normal.y, vec_points[ii].normal.z);
                        double y_min = -m_config.groundSeg.ground_max_z;
                        double y_max = -m_config.groundSeg.ground_min_z;
                        if (y_min < p3.y() && p3.y() < y_max) {
                            cloud_cut_ground->points_.push_back(p3);
                            cloud_cut_ground->colors_.push_back(Eigen::Vector3d(1.0,0.0,0.0));
                            cloud_cut_ground->normals_.push_back(n3);
                            vec_cloud_cut.push_back(p3.x());
                            vec_cloud_cut.push_back(p3.y());
                            vec_cloud_cut.push_back(p3.z());
                        } else {
                            cloud_cut_nonground->points_.push_back(p3);
                            cloud_cut_nonground->colors_.push_back(Eigen::Vector3d(item.color.red/255.0, item.color.green/255.0, item.color.blue/255.0));
                            cloud_cut_nonground->normals_.push_back(n3);
                        }
                    }
                    Groud_seg groud_seg;
                    std::vector<float> out_ground_cloud, out_non_ground_cloud;
                    groud_seg.groud_seg_my(vec_cloud_cut, out_ground_cloud, out_non_ground_cloud, "");
                    for (size_t j = 0; j < out_ground_cloud.size()/3; ++j) {
                        Eigen::Vector3d p(out_ground_cloud[3*j], out_ground_cloud[3*j+1], out_ground_cloud[3*j+2]);
                        pc_ground->points_.push_back(p);
                        pc_ground->colors_.push_back(Eigen::Vector3d(1.0,0.0,0.0));
                    }
                    for (size_t j = 0; j < out_non_ground_cloud.size()/3; ++j) {
                        Eigen::Vector3d p(out_non_ground_cloud[3*j], out_non_ground_cloud[3*j+1], out_non_ground_cloud[3*j+2]);
                        pc_nonground->points_.push_back(p);
                        pc_nonground->colors_.push_back(Eigen::Vector3d(item.color.red/255.0, item.color.green/255.0, item.color.blue/255.0));
                    }
                    pc_nonground->points_.insert(pc_nonground->points_.end(),
                                                 cloud_cut_nonground->points_.begin(), cloud_cut_nonground->points_.end());
                    if (!cloud_cut_nonground->colors_.empty())
                        pc_nonground->colors_.insert(pc_nonground->colors_.end(),
                                                     cloud_cut_nonground->colors_.begin(), cloud_cut_nonground->colors_.end());
                    if (!cloud_cut_nonground->normals_.empty())
                        pc_nonground->normals_.insert(pc_nonground->normals_.end(),
                                                      cloud_cut_nonground->normals_.begin(), cloud_cut_nonground->normals_.end());
                    stats.ground_ms += (now_ms() - t_ground_begin);
                } else {
                    for (size_t ii = 0; ii < vec_points.size(); ++ii) {
                        Eigen::Vector3d p3(vec_points[ii].position.x, vec_points[ii].position.y, vec_points[ii].position.z);
                        pc_nonground->points_.push_back(p3);
                        pc_nonground->colors_.push_back(Eigen::Vector3d(item.color.red/255.0, item.color.green/255.0, item.color.blue/255.0));
                    }
                }

                // ground 输出（转换到世界坐标后放入 local_ground）
                if (m_config.fdageParam.output_map && pc_ground && !pc_ground->points_.empty()) {
                    ufo::PointCloudColor cloud_ground_tmp(pc_ground->points_.size());
                    for (size_t i = 0; i < pc_ground->points_.size(); ++i) {
                        const auto &pg = pc_ground->points_[i];
                        cloud_ground_tmp[i].x = pg.x();
                        cloud_ground_tmp[i].y = pg.y();
                        cloud_ground_tmp[i].z = pg.z();
                        cloud_ground_tmp[i].red = 255;
                        cloud_ground_tmp[i].green = 0;
                        cloud_ground_tmp[i].blue = 0;
                    }
                    ufo::applyTransform(cloud_ground_tmp, item.sweep.pose6f);
                    local_ground.insert(local_ground.end(), cloud_ground_tmp.begin(), cloud_ground_tmp.end());
                }

                // 非地面向量化 + near-camera + seenFree 分类
                // ===== 可选体素下采样（减小后续分类与 seenFree 查询体量） =====
                size_t nn = pc_nonground->points_.size();
                if (nn > 2000) { // 简单阈值避免小云开销
                    double t_ds_begin = now_ms();
                    // 动态 voxel: 基础为分辨率 * 2，确保不低于 1e-3
                    double voxel_size = std::max(1e-3, m_config.map.resolution * 2.0);
                    // 对于点数极大时进一步增大体素（分段）
                    if (nn > 50000) voxel_size *= 1.5;
                    if (nn > 150000) voxel_size *= 2.0;
                    auto pc_ds = pc_nonground->VoxelDownSample(voxel_size);
                    if (pc_ds && !pc_ds->points_.empty() && pc_ds->points_.size() < nn) {
                        pc_nonground = pc_ds;
                        nn = pc_nonground->points_.size();
                    }
                    stats.downsample_ms += (now_ms() - t_ds_begin);
                }
                stats.points_after_ds += nn;
                double t_classify_begin = now_ms();
                Eigen::Matrix<double,3,Eigen::Dynamic> N(3, nn);
                double t_build_matrix_begin = now_ms();
                for (size_t i = 0; i < nn; ++i) {
                    const auto &p = pc_nonground->points_[i];
                    N(0,i) = p.x(); N(1,i) = p.y(); N(2,i) = p.z();
                }
                // 包括 R/T、TN 计算在 build_points_ms 扩展统计中更细分也可，这里合并
                stats.build_points_ms += (now_ms() - t_build_matrix_begin);
                auto r = item.sweep.pose6f.rotation.rotMatrix();
                auto t = item.sweep.pose6f.translation;
                Eigen::Matrix3d R; R << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
                Eigen::Vector3d T(t.x, t.y, t.z);
                Eigen::Matrix<double,3,Eigen::Dynamic> TN = (R * N).colwise() + T;

                cv::Mat pointsB((int)nn, 3, CV_32F);
                std::vector<int> select_idx; int count = 0;
                // 缩窄 near-camera y-window: [-0.15, 0.15] 而不是 [-0.2,0.2]
                for (size_t i = 0; i < nn; ++i) {
                    double wy = TN(1,i);
                    if (wy > -0.15 && wy < 0.15) {
                        select_idx.push_back((int)i);
                        pointsB.at<float>(count,0) = (float)TN(0,i);
                        pointsB.at<float>(count,1) = (float)TN(1,i);
                        pointsB.at<float>(count,2) = (float)TN(2,i);
                        ++count;
                    }
                }
                pointsB = pointsB.rowRange(0,count).clone();
                std::vector<char> near_cam(nn,0);
                if (pointsB.rows > 0) {
                    cv::Mat indices,dists;
                    double t_flann_begin = now_ms();
                    localFlann.knnSearch(pointsB, indices, dists, 1, cv::flann::SearchParams(64));
                    stats.classify_flann_ms += (now_ms() - t_flann_begin);
                    float maxDistSq = 0.3f * 0.3f; // 比较平方距离避免 sqrt
                    for (int r_i = 0; r_i < pointsB.rows; ++r_i) {
                        float dist_sq = dists.at<float>(r_i,0); // FLANN 返回的是平方距离
                        if (dist_sq <= maxDistSq) near_cam[select_idx[r_i]] = 1;
                    }
                }

                for (size_t i = 0; i < nn; ++i) {
                    ufo::PointCloudColor tmp(1);
                    tmp[0].x = TN(0,i); tmp[0].y = TN(1,i); tmp[0].z = TN(2,i);
                    if (!pc_nonground->colors_.empty()) {
                        auto col = pc_nonground->colors_[i];
                        tmp[0].red   = (unsigned char)std::clamp(int(std::round(col.x()*255.0)),0,255);
                        tmp[0].green = (unsigned char)std::clamp(int(std::round(col.y()*255.0)),0,255);
                        tmp[0].blue  = (unsigned char)std::clamp(int(std::round(col.z()*255.0)),0,255);
                    } else {
                        tmp[0].red = tmp[0].green = tmp[0].blue = 255;
                    }
                    bool remove_flag = near_cam[i];
                    double t_seen_begin = now_ms();
                    if (!remove_flag && !map.seenFree(tmp[0])) {
                        local_static.push_back(tmp[0]);
                    } else {
                        tmp[0].red = 255; tmp[0].green = 0; tmp[0].blue = 0;
                        local_remove.push_back(tmp[0]);
                    }
                    stats.classify_seenfree_ms += (now_ms() - t_seen_begin);
                }
                stats.classify_ms += (now_ms() - t_classify_begin);
                stats.sweeps++;

                size_t cur = ++processed;
                if (!m_config.printing.verbose) {
                    std::lock_guard<std::mutex> lk(cout_mutex);
                    std::cout << "(" << cur << "/" << vec_sweepLocation.size() << ") Processing: " << item.sweep.uuid << "\r" << std::flush;
                }

                if (m_config.fdageParam.replaceInputPly) {
                    Potree::AABB dummy_aabb;
                    std::string las_out = m_new_laz_dir + std::to_string(item.sweep.floor_id) + "_" + item.sweep.uuid + "_depth.laz";
                    ufo::writePointCloudLAS(las_out, vec_points, dummy_aabb);
                }
            } // while pop

            // 合并本地结果
            if (!local_static.empty() || !local_remove.empty() || !local_ground.empty()) {
                std::lock_guard<std::mutex> lk(results_mutex);
                if (!local_static.empty()) cloud_static.insert(cloud_static.end(), local_static.begin(), local_static.end());
                if (!local_remove.empty()) cloud_remove.insert(cloud_remove.end(), local_remove.begin(), local_remove.end());
                if (!local_ground.empty()) cloud_acc_ground.insert(cloud_acc_ground.end(), local_ground.begin(), local_ground.end());
                // 将线程统计累加到全局（复用 cloud_acc_noise 作为临时，不合适，改用静态 vector）
            }
            // 将统计写入一个全局数组（需提前定义）
            {
                std::lock_guard<std::mutex> lk(results_mutex);
                thread_stats_all.push_back(stats);
            }
        });
    }

    producer.join();
    for (auto &th : workers) th.join();

    // 后处理与输出
    // 统一 propagate（前面已强制传 false）
    map.propagateModified();
    if (m_config.clustering.cluster) cluster(map, m_config.clustering);
    auto end_t = high_resolution_clock::now();
    double seconds = duration_cast<milliseconds>(end_t - begin_t).count() / 1000.0;
    std::cout << "Prefetch pipeline cost: " << seconds << "s\n";

    // 输出线程统计
    if (!thread_stats_all.empty()) {
        double io_sum=0, insert_sum=0, ground_sum=0, classify_sum=0; size_t sweeps_total=0; size_t pts_total=0;
        double ds_sum=0; size_t pts_after_total=0; double lock_wait_sum=0; double call_sum=0; double flann_sum=0; double seen_sum=0; double transform_sum=0; double build_sum=0; double queue_wait_sum=0;
        for (auto &st : thread_stats_all) {
            io_sum += st.io_ms; insert_sum += st.insert_ms; ground_sum += st.ground_ms; classify_sum += st.classify_ms; ds_sum += st.downsample_ms; sweeps_total += st.sweeps; pts_total += st.points_read; pts_after_total += st.points_after_ds; lock_wait_sum += st.insert_lock_wait_ms; call_sum += st.insert_call_ms; flann_sum += st.classify_flann_ms; seen_sum += st.classify_seenfree_ms; transform_sum += st.transform_ms; build_sum += st.build_points_ms; queue_wait_sum += st.queue_wait_ms;
        }
        std::cout << "[Perf] sweeps=" << sweeps_total << " points_in=" << pts_total << " points_after_ds=" << pts_after_total << '\n'
                  << "[Perf] io(ms)=" << io_sum << " insert(ms)=" << insert_sum << " (lock_wait=" << lock_wait_sum << ", call=" << call_sum << ")"
                  << " ground(ms)=" << ground_sum << " downsample(ms)=" << ds_sum << " classify(ms)=" << classify_sum
                  << " (flann=" << flann_sum << ", seenFree=" << seen_sum << ")" << '\n'
                  << "[Perf] transform(ms)=" << transform_sum << " build_points(ms)=" << build_sum << " queue_wait(ms)=" << queue_wait_sum << '\n'
                  << "[Perf] avg_per_sweep(ms): io=" << (io_sum/(sweeps_total? sweeps_total:1))
                  << " insert=" << (insert_sum/(sweeps_total? sweeps_total:1))
                  << " ground=" << (ground_sum/(sweeps_total? sweeps_total:1))
                  << " downsample=" << (ds_sum/(sweeps_total? sweeps_total:1))
                  << " classify=" << (classify_sum/(sweeps_total? sweeps_total:1)) << std::endl;
    }
    if (m_config.fdageParam.output_map) {
        ufo::Color color_remove_final{255,0,0};
        writePointCloudPLY_downsample_remove(m_data_root + (m_config.output.filename + "_remove_prefetch.ply"), cloud_remove, 0.1,0.1,0.1, color_remove_final);
        ufo::writePointCloudPLY_downsample(m_data_root + (m_config.output.filename + "_static_prefetch.ply"), cloud_static, 0.1,0.1,0.1);
        ufo::writePointCloudPLY_downsample(m_data_root + (m_config.output.filename + "_ground_prefetch.ply"), cloud_acc_ground, 0.1,0.1,0.1);
        ufo::writePointCloudPLY_downsample(m_data_root + (m_config.output.filename + "_camera_traj_prefetch.ply"), cloud_camera_traj, 0.01,0.01,0.01);
    }

}

void CleanMap::run(){
    auto begin_t = high_resolution_clock::now();

    cv::Mat pointsA(m_map_id_sweepLocation.size(), 3, CV_32F);
    ufo::PointCloudColor cloud_camera_traj;

    get_camera_traj(cloud_camera_traj, pointsA);
    // pointsA 用于构造每线程的 flann 索引（线程本地或共享，下面使用每线程本地）
    //cv::flann::Index globalFlann(pointsA, cv::flann::KDTreeIndexParams(4));
    
    ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION | ufo::MapType::LABEL> map(
        m_config.map.resolution, m_config.map.levels);
    map.reserve(100'000'000);

    Potree::AABB all_aabb;
    std::mutex mtx;

    int num_location = m_map_id_sweepLocation.size();
    ufo::Timing timing;
    timing.start("Total");

    // 为并行区准备 SweepLocation 向量，避免并行中访问 map 容器
    std::vector<SweepLocation> vec_sweepLocation;
    vec_sweepLocation.reserve(m_map_id_sweepLocation.size());
    for(auto& pair : m_map_id_sweepLocation){
        vec_sweepLocation.push_back(pair.second);
    }

    // 为并行区准备每线程 RNG 种子（线程安全）
    std::random_device rd;
    std::mt19937 seed_gen(rd());
    int omp_threads = omp_get_max_threads();
    std::vector<unsigned int> thread_seeds(omp_threads);
    for (int ti = 0; ti < omp_threads; ++ti) thread_seeds[ti] = seed_gen();

    // 用于并行任务进度的原子计数器（替代共享 i++)
    std::atomic<int> progress_counter{0};

    // 全局输出容器（并行内使用本地缓冲，最终合并）
    ufo::PointCloudColor cloud_static;
    ufo::PointCloudColor cloud_remove;
    ufo::PointCloudColor cloud_acc_ground;
    ufo::PointCloudColor cloud_acc_noise;

#ifdef _OPENMP
    std::cout << "use openmp, waiting..." << std::endl;
#endif

    // 合并两段并行逻辑为一次并行区
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // 每线程本地 RNG（避免共享 std::mt19937）
        std::mt19937 local_gen(thread_seeds[tid]);
        std::uniform_int_distribution<> local_dis(0, 255);

        // 每线程本地累积容器，减少锁竞争
        ufo::PointCloudColor local_cloud_static;
        ufo::PointCloudColor local_cloud_remove;
        ufo::PointCloudColor local_cloud_acc_ground;
        ufo::PointCloudColor local_cloud_acc_noise;

        // 每线程本地 flann index（避免不确定的线程安全问题）
        cv::flann::Index localFlann(pointsA, cv::flann::KDTreeIndexParams(4));

        #pragma omp for schedule(dynamic, 1)
        for (int loc_index = 0; loc_index < (int)vec_sweepLocation.size(); ++loc_index) {
            // 复制 SweepLocation
            SweepLocation ele = vec_sweepLocation[loc_index];
            std::string las_path = ele.model_path;
            ufo::Pose6f viewpoint = ele.pose6f;

            // 进度计数（原子）
            int cur = progress_counter.fetch_add(1) + 1;
            if (cur > 1 && !m_config.printing.verbose) {
                std::ostringstream log_msg;
                log_msg << "(" << cur << "/" << num_location << ") Processing: " << ele.uuid << " Time Cost: "
                        << m_config.integration.timing.lastSeconds() << "s";
                std::string spaces(10, ' ');
                log_msg << spaces;
                {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "\r" << log_msg.str() << std::flush;
                }
            }

            // 线程本地随机着色
            ufo::Color color;
            color.red = local_dis(local_gen);
            color.green = local_dis(local_gen);
            color.blue = local_dis(local_gen);

            // 读取点云并保留一份未变换的原始副本
            ufo::PointCloudColor cloud;
            Potree::AABB aabb;
            ufo::readPointCloudLAZ(las_path, cloud, aabb, color);
            // 保存原始点云（未应用 viewpoint 变换），以便后续需要原始坐标的处理
            ufo::PointCloudColor cloud_raw = cloud;
            // 对 cloud 应用变换用于构建地图和基于世界坐标的分析
            ufo::applyTransform(cloud, viewpoint);

            // 更新全局 aabb
            {
                std::lock_guard<std::mutex> guard(mtx);
                all_aabb.update(aabb);
            }

            // 插入到地图（保持原来对 insertPointCloud 的互斥）
            {
                std::lock_guard<std::mutex> guard(mtx);
                ufo::insertPointCloud(map, cloud, viewpoint.translation, m_config.integration, m_config.propagate);
            }

            // 以下为原第二段循环中对同一位置的后续分析逻辑

            // 注意：vec_points 需要使用原始的点云坐标（文件坐标系），
            // 因此这里使用 cloud_raw 而不是已经被变换到世界坐标的 cloud。
            size_t numPoints = cloud_raw.size();
            std::vector<Potree::Point> vec_points;
            vec_points.resize(numPoints);
            for (size_t i = 0; i < numPoints; ++i) {
                Potree::Point point;
                point.position.x = cloud_raw[i].x;
                point.position.y = cloud_raw[i].y;
                point.position.z = cloud_raw[i].z;

                point.color.x = cloud_raw[i].red;
                point.color.y = cloud_raw[i].green;
                point.color.z = cloud_raw[i].blue;

                vec_points[i] = point;
            }

            // open3d 替代 PCL: pc_nonground / pc_ground
            auto pc_nonground = std::make_shared<open3d::geometry::PointCloud>();
            auto pc_ground = std::make_shared<open3d::geometry::PointCloud>();

            bool bfilterGroundPlane = m_config.groundSeg.enable;

            if (bfilterGroundPlane) {
                auto cloud_cut_ground = std::make_shared<open3d::geometry::PointCloud>();
                auto cloud_cut_nonground = std::make_shared<open3d::geometry::PointCloud>();
                std::vector<float> vec_cloud_cut;

                for (size_t ii = 0; ii < vec_points.size(); ++ii) {
                    Eigen::Vector3d p3(vec_points[ii].position.x, vec_points[ii].position.y, vec_points[ii].position.z);
                    Eigen::Vector3d c3(color.red/255.0, color.green/255.0, color.blue/255.0);
                    Eigen::Vector3d n3(vec_points[ii].normal.x, vec_points[ii].normal.y, vec_points[ii].normal.z);

                    double y_min = -m_config.groundSeg.ground_max_z;
                    double y_max = -m_config.groundSeg.ground_min_z;

                    if (y_min < p3.y() && p3.y() < y_max) {
                        cloud_cut_ground->points_.push_back(p3);
                        cloud_cut_ground->colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
                        cloud_cut_ground->normals_.push_back(n3);

                        vec_cloud_cut.push_back(vec_points[ii].position.x);
                        vec_cloud_cut.push_back(vec_points[ii].position.y);
                        vec_cloud_cut.push_back(vec_points[ii].position.z);
                    } else {
                        cloud_cut_nonground->points_.push_back(p3);
                        cloud_cut_nonground->colors_.push_back(c3);
                        cloud_cut_nonground->normals_.push_back(n3);
                    }
                }

                // 调用已有的 ground segmentation
                Groud_seg groud_seg = Groud_seg();
                std::vector<float> out_ground_cloud;
                std::vector<float> out_non_ground_cloud;
                groud_seg.groud_seg_my(vec_cloud_cut, out_ground_cloud, out_non_ground_cloud, "");

                for (size_t j = 0; j < out_ground_cloud.size()/3; ++j) {
                    Eigen::Vector3d p(out_ground_cloud[3*j], out_ground_cloud[3*j + 1], out_ground_cloud[3*j + 2]);
                    pc_ground->points_.push_back(p);
                    pc_ground->colors_.push_back(Eigen::Vector3d(1.0,0.0,0.0));
                }

                for (size_t j = 0; j < out_non_ground_cloud.size()/3; ++j) {
                    Eigen::Vector3d p(out_non_ground_cloud[3*j], out_non_ground_cloud[3*j + 1], out_non_ground_cloud[3*j + 2]);
                    pc_nonground->points_.push_back(p);
                    pc_nonground->colors_.push_back(Eigen::Vector3d(color.red/255.0, color.green/255.0, color.blue/255.0));
                }

                // append cut nonground
                pc_nonground->points_.insert(pc_nonground->points_.end(),
                                             cloud_cut_nonground->points_.begin(), cloud_cut_nonground->points_.end());
                if (!cloud_cut_nonground->colors_.empty())
                    pc_nonground->colors_.insert(pc_nonground->colors_.end(),
                                                 cloud_cut_nonground->colors_.begin(), cloud_cut_nonground->colors_.end());
                if (!cloud_cut_nonground->normals_.empty())
                    pc_nonground->normals_.insert(pc_nonground->normals_.end(),
                                                  cloud_cut_nonground->normals_.begin(), cloud_cut_nonground->normals_.end());

                
                // 若需要输出 ground，构造并累积到线程本地 ground 容器
                if (pc_ground->points_.size() > 0 && m_config.fdageParam.output_map) {

                    ufo::PointCloudColor cloud_ground_tmp;
                    cloud_ground_tmp.resize(pc_ground->points_.size());

                    for (size_t ii = 0; ii < pc_ground->points_.size(); ++ii) {
                        const auto &p = pc_ground->points_[ii];
                        cloud_ground_tmp[ii].x = p.x();
                        cloud_ground_tmp[ii].y = p.y();
                        cloud_ground_tmp[ii].z = p.z();
                        cloud_ground_tmp[ii].red   = 255;
                        cloud_ground_tmp[ii].green = 0;
                        cloud_ground_tmp[ii].blue  = 0;
                    }
                    std::lock_guard<std::mutex> guard(mtx);
                    ufo::applyTransform(cloud_ground_tmp, viewpoint);
                    local_cloud_acc_ground.insert(local_cloud_acc_ground.end(), cloud_ground_tmp.cbegin(), cloud_ground_tmp.cend());
                }
            } else {
                for (size_t ii = 0; ii < vec_points.size(); ++ii) {
                    Eigen::Vector3d p3(vec_points[ii].position.x, vec_points[ii].position.y, vec_points[ii].position.z);
                    pc_nonground->points_.push_back(p3);
                    pc_nonground->colors_.push_back(Eigen::Vector3d(color.red/255.0, color.green/255.0, color.blue/255.0));
                    pc_nonground->normals_.push_back(Eigen::Vector3d(vec_points[ii].normal.x, vec_points[ii].normal.y, vec_points[ii].normal.z));
                }
            }

            // transforms
            auto r = viewpoint.rotation.rotMatrix();
            auto t =  viewpoint.translation;
            ufo::PointCloudColor pt;
            pt.resize(1);
            // ===== Eigen 向量化: 预计算非地面点世界坐标 =====
            Eigen::Matrix3d R_vec;
            R_vec << r[0], r[1], r[2],
                      r[3], r[4], r[5],
                      r[6], r[7], r[8];
            Eigen::Vector3d T_vec(t.x, t.y, t.z);
            size_t nongroundN_vec = pc_nonground->points_.size();
            Eigen::Matrix<double,3,Eigen::Dynamic> N_vec(3, nongroundN_vec);
            for (size_t i_ng = 0; i_ng < nongroundN_vec; ++i_ng) {
                const auto &p_ng = pc_nonground->points_[i_ng];
                N_vec(0,i_ng) = p_ng.x();
                N_vec(1,i_ng) = p_ng.y();
                N_vec(2,i_ng) = p_ng.z();
            }
            Eigen::Matrix<double,3,Eigen::Dynamic> TN_vec = (R_vec * N_vec).colwise() + T_vec;

            std::vector<Potree::Point> vec_points_static;
            std::vector<Potree::Point> vec_points_remove;
            Potree::AABB static_point_aabb;

            for(size_t ip = 0; ip < pc_ground->points_.size(); ++ip){
                const auto &p = pc_ground->points_[ip];
                Eigen::Vector3d normal = pc_ground->normals_.empty() ? Eigen::Vector3d(0,0,0) : pc_ground->normals_[ip];

                Potree::Point point;
                point.position.x = p.x();
                point.position.y = p.y();
                point.position.z = p.z();

                point.normal.x = normal.x();
                point.normal.y = normal.y();
                point.normal.z = normal.z();

                point.color.x = 255;
                point.color.y = 255;
                point.color.z = 0;

                vec_points_static.push_back(point);	

                Potree::Vector3<double> pt_tmp(p.x(), p.y(), p.z());
                static_point_aabb.update(pt_tmp);

                if(m_config.fdageParam.output_map){
                    std::lock_guard<std::mutex> guard(mtx);
                    pt[0].x   = r[0] * p.x() + r[1] * p.y() + r[2] * p.z() + t.x;
                    pt[0].y   = r[3] * p.x() + r[4] * p.y() + r[5] * p.z() + t.y;
                    pt[0].z   = r[6] * p.x() + r[7] * p.y() + r[8] * p.z() + t.z;

                    pt[0].red = 255;
                    pt[0].green = 255;
                    pt[0].blue = 0;
                    local_cloud_static.push_back(pt[0]);   
			    }
            }

            // use camera traj + flann (localFlann) to mark near-camera points -> mark remove vs keep
            {
                cv::Mat pointsB(static_cast<int>(pc_nonground->points_.size()), 3, CV_32F);
                int count = 0;
                std::vector<int> vec_select_index;
                for (size_t pi = 0; pi < pc_nonground->points_.size(); ++pi) {
                    const auto &pp = pc_nonground->points_[pi];
                    double x = pp.x();
                    double y = pp.y();
                    double z = pp.z();

                    pt[0].x = r[0] * x + r[1] * y + r[2] * z + t.x;
                    pt[0].y = r[3] * x + r[4] * y + r[5] * z + t.y;
                    pt[0].z = r[6] * x + r[7] * y + r[8] * z + t.z;

                    if (pt[0].y > -0.2 && pt[0].y < 0.2) {
                        vec_select_index.push_back(static_cast<int>(pi));
                        pointsB.at<float>(count, 0) = static_cast<float>(pt[0].x);
                        pointsB.at<float>(count, 1) = static_cast<float>(pt[0].y);
                        pointsB.at<float>(count, 2) = static_cast<float>(pt[0].z);
                        ++count;
                    }
                }
                pointsB = pointsB.rowRange(0, count).clone();

                if (pointsB.rows > 0) {
                    cv::Mat indices, dists;
                    localFlann.knnSearch(pointsB, indices, dists, 1, cv::flann::SearchParams(64));
                    //globalFlann.knnSearch(pointsB, indices, dists, 1, cv::flann::SearchParams(64));

                    std::vector<int> vec_label(static_cast<int>(pc_nonground->points_.size()), 0);
                    float maxDistance = 0.3f;
                    for (int pt_i = 0; pt_i < pointsB.rows; ++pt_i) {
                        float distance = std::sqrt(dists.at<float>(pt_i, 0));
                        if (distance <= maxDistance) vec_label[vec_select_index[pt_i]] = 1;
                    }

                    auto pc_nonground_tmp = std::make_shared<open3d::geometry::PointCloud>();
                    for (size_t j = 0; j < vec_label.size(); ++j) {
                        if (vec_label[j] == 1) {
                            //ufo::PointCloudColor tmp_pt(1);
                            const auto &p = pc_nonground->points_[j];

                            double x = p.x();
                            double y = p.y();
                            double z = p.z();

                            pt[0].x = r[0] * x + r[1] * y + r[2] * z + t.x;
                            pt[0].y = r[3] * x + r[4] * y + r[5] * z + t.y;
                            pt[0].z = r[6] * x + r[7] * y + r[8] * z + t.z;

                            local_cloud_remove.push_back(pt[0]);

                        } else {
                            pc_nonground_tmp->points_.push_back(pc_nonground->points_[j]);
                            if (!pc_nonground->colors_.empty()) pc_nonground_tmp->colors_.push_back(pc_nonground->colors_[j]);
                            if (!pc_nonground->normals_.empty()) pc_nonground_tmp->normals_.push_back(pc_nonground->normals_[j]);
                        }
                    }
                    pc_nonground = pc_nonground_tmp;
                }
            }

            // 近相机过滤可能修改 pc_nonground，重新构建向量化矩阵
            nongroundN_vec = pc_nonground->points_.size();
            N_vec.resize(3, nongroundN_vec);
            for (size_t i_ng2 = 0; i_ng2 < nongroundN_vec; ++i_ng2) {
                const auto &p_ng2 = pc_nonground->points_[i_ng2];
                N_vec(0,i_ng2) = p_ng2.x();
                N_vec(1,i_ng2) = p_ng2.y();
                N_vec(2,i_ng2) = p_ng2.z();
            }
            TN_vec = (R_vec * N_vec).colwise() + T_vec;
            // 非地面点 (向量化坐标 TN_vec) -> 判断 map.seenFree
            for (size_t idx2 = 0; idx2 < nongroundN_vec; ++idx2) {
                ufo::PointCloudColor tmp_elem(1);
                tmp_elem[0].x = TN_vec(0, idx2);
                tmp_elem[0].y = TN_vec(1, idx2);
                tmp_elem[0].z = TN_vec(2, idx2);
                if (!pc_nonground->colors_.empty()) {
                    auto col = pc_nonground->colors_[idx2];
                    tmp_elem[0].red   = (unsigned char)std::clamp(int(std::round(col.x() * 255.0)), 0, 255);
                    tmp_elem[0].green = (unsigned char)std::clamp(int(std::round(col.y() * 255.0)), 0, 255);
                    tmp_elem[0].blue  = (unsigned char)std::clamp(int(std::round(col.z() * 255.0)), 0, 255);
                } else {
                    tmp_elem[0].red = tmp_elem[0].green = tmp_elem[0].blue = 255;
                }
                if (!map.seenFree(tmp_elem[0])) {
                    local_cloud_static.push_back(tmp_elem[0]);
                } else {
                    tmp_elem[0].red = 255; tmp_elem[0].green = 0; tmp_elem[0].blue = 0;
                    local_cloud_remove.push_back(tmp_elem[0]);
                }
            }

            // 可选写替换输入文件（保持原行为）
            if (m_config.fdageParam.replaceInputPly) {
                int floor_id = ele.floor_id;
                std::string las_out = m_new_laz_dir + std::to_string(floor_id) + "_" + ele.uuid + "_depth.laz";
                ufo::writePointCloudLAS(las_out, vec_points, static_point_aabb); // 保持原意，静态 aabb 可根据需要调整
            }
        } // end for
        // 合并线程本地累积到全局累积（加锁）
        {
            std::lock_guard<std::mutex> guard(mtx);
            if (!local_cloud_static.empty())
                cloud_static.insert(cloud_static.end(), local_cloud_static.cbegin(), local_cloud_static.cend());
            if (!local_cloud_remove.empty())
                cloud_remove.insert(cloud_remove.end(), local_cloud_remove.cbegin(), local_cloud_remove.cend());
            if (!local_cloud_acc_ground.empty())
                cloud_acc_ground.insert(cloud_acc_ground.end(), local_cloud_acc_ground.cbegin(), local_cloud_acc_ground.cend());
            if (!local_cloud_acc_noise.empty())
                cloud_acc_noise.insert(cloud_acc_noise.end(), local_cloud_acc_noise.cbegin(), local_cloud_acc_noise.cend());
        }
    } // end parallel

    // 建图全部时间
    auto build_map_end = high_resolution_clock::now();
    long long duration_map = duration_cast<milliseconds>(build_map_end - begin_t).count();
    float seconds_map = duration_map / 1'000.0f;
    std::cout << "build+analysis cost: " << seconds_map << "s" << std::endl;

    // 把合并后的结果用于后续流程（保留原来逻辑写文件 / 聚类 / 输出）
    cloud_acc_noise = std::move(cloud_acc_noise);

    indicators::show_console_cursor(true);
    std::cout << "\033[0m\n[LOG] Step 3: Finished Processing data. Start saving map... " << std::endl;

    if (!m_config.propagate) {
        timing.setTag("Propagate");
        map.propagateModified();
    }

    if (m_config.clustering.cluster) {
        cluster(map, m_config.clustering);
    }

    // 流程全部时间
    auto end = high_resolution_clock::now();
    long long duration = duration_cast<milliseconds>(end - begin_t).count();
    float seconds = duration / 1'000.0f;
    std::cout << "cost total time: " << seconds << "s" << std::endl;

    if(m_config.fdageParam.output_map)
    {
        std::cout << "output map begin" << std::endl;
        ufo::Color color_remove;
        color_remove.red = 255;
        color_remove.green = 0;
        color_remove.blue = 0;
        std::string save_remove_name = "_remove.ply";
        std::string save_static_name = "_static.ply";
        std::string save_ground_name = "_ground.ply";
        std::string save_noise_name = "_noise.ply";

        writePointCloudPLY_downsample_remove(m_data_root +  (m_config.output.filename + save_remove_name ), cloud_remove, 0.1, 0.1, 0.1, color_remove);
        ufo::writePointCloudPLY_downsample(m_data_root +  (m_config.output.filename + save_static_name), cloud_static, 0.1, 0.1, 0.1);
        ufo::writePointCloudPLY_downsample(m_data_root +  (m_config.output.filename + save_ground_name), cloud_acc_ground, 0.1, 0.1, 0.1);
        ufo::writePointCloudPLY_downsample(m_data_root +  (m_config.output.filename + save_noise_name), cloud_acc_noise, 0.1, 0.1, 0.1);

        std::string save_camera_traj = "_camera_traj.ply";
        ufo::writePointCloudPLY_downsample(m_data_root +  (m_config.output.filename + save_camera_traj), cloud_camera_traj, 0.01, 0.01, 0.01);

        std::cout << "output map end" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    std::string data_root = argv[1];
	std::string config_file_path = argv[2];

	 CleanMap clean_map = CleanMap(data_root, config_file_path);
     clean_map.init();
	 clean_map.run_prefetch();

	 return 0;
}

