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

// TOML
#include "toml.hpp"
#include "indicators.hpp"

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

#include <pcl/common/transforms.h>
//#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>

struct SweepLocation {
	int floor_id;
	std::string uuid;
	ufo::Pose6f pose6f;
	std::string model_path;
};

struct Dataset {
	std::size_t first = 0;
	std::size_t last  = -1;
	std::size_t num   = -1;
};

struct Map {
	ufo::node_size_t resolution = 0.1;  // In meters
	ufo::depth_t     levels     = 17;   // Levels of the octree
};

struct Clustering {
	bool         cluster      = false;
	float        max_distance = 1.2f;
	std::size_t  min_points   = 1000;
	ufo::depth_t depth        = 0;
};

struct Printing {
	bool verbose = true;
	bool debug   = false;
};

struct Output {
	std::string filename   = "dufomap";
	bool        has_color  = false;
	bool        raycasting = false;
	bool        voxel_map  = false;
};

struct GroundSeg {
	bool enable = true;
	float distance = 0.04;
	float angle = 0.15;
	float planeDistance = 0.15;
	float ground_min_z=-1.7;
	float ground_max_z=-0.5;
};

struct FilterNoise {
	bool enable = true;
	int filterMeanK = 50;
	float StddevMulThresh = 1.0;
};

struct Fdage {
	bool replaceInputPly=false;
	bool output_map=false;
	int select_m = -1;
	int select_n = -1;
};

struct Config {
	Dataset                dataset;
	Map                    map;
	ufo::IntegrationParams integration;
	bool                   propagate = false;
	Clustering             clustering;
	Printing               printing;
	Output                 output;

	GroundSeg      groundSeg;
	FilterNoise      filterNoise;
	Fdage                fdageParam;

	void read(toml::table tbl)
	{
		map.resolution = read(tbl["important"]["resolution"], map.resolution);

		dataset.first = read(tbl["dataset"]["first"], dataset.first);
		dataset.last  = read(tbl["dataset"]["last"], dataset.last);
		dataset.num   = read(tbl["dataset"]["num"], dataset.num);
		map.levels     = read(tbl["map"]["levels"], map.levels);

		auto dsm = read(tbl["integration"]["down_sampling_method"], std::string("none"));
		integration.down_sampling_method =
		    "none" == dsm
		        ? ufo::DownSamplingMethod::NONE
		        : ("centroid" == dsm ? ufo::DownSamplingMethod::CENTROID
		                             : ("uniform" == dsm ? ufo::DownSamplingMethod::UNIFORM
		                                                 : ufo::DownSamplingMethod::CENTER));
		integration.hit_depth = read(tbl["integration"]["hit_depth"], integration.hit_depth);
		integration.miss_depth =
		    read(tbl["integration"]["miss_depth"], integration.miss_depth);
		integration.min_range = read(tbl["integration"]["min_range"], integration.min_range);
		integration.max_range = read(tbl["integration"]["max_range"], integration.max_range);
		integration.inflate_unknown =
		    read(tbl["important"]["inflate_unknown"], integration.inflate_unknown);
		integration.inflate_unknown_compensation =
		    read(tbl["integration"]["inflate_unknown_compensation"],
		         integration.inflate_unknown_compensation);
		integration.ray_passthrough_hits = read(tbl["integration"]["ray_passthrough_hits"],
		                                        integration.ray_passthrough_hits);
		integration.inflate_hits_dist =
		    read(tbl["important"]["inflate_hits_dist"], integration.inflate_hits_dist);
		integration.parallel = tbl["integration"]["parallel"].value_or(integration.parallel);
		integration.num_threads =
		    read(tbl["integration"]["num_threads"], integration.num_threads);
		integration.only_valid =
		    read(tbl["integration"]["only_valid"], integration.only_valid);

		propagate = read(tbl["integration"]["propagate"], propagate);

		clustering.cluster = read(tbl["clustering"]["cluster"], clustering.cluster);
		clustering.max_distance =
		    read(tbl["clustering"]["max_distance"], clustering.max_distance);
		clustering.min_points = read(tbl["clustering"]["min_points"], clustering.min_points);
		clustering.depth      = read(tbl["clustering"]["depth"], clustering.depth);

		printing.verbose = read(tbl["printing"]["verbose"], printing.verbose);
		printing.debug   = read(tbl["printing"]["debug"], printing.debug);

		output.filename   = read(tbl["output"]["filename"], output.filename);
		output.has_color  = read(tbl["output"]["has_color"], output.has_color);
		output.raycasting = read(tbl["output"]["raycasting"], output.raycasting);
		output.voxel_map = read(tbl["output"]["voxel_map"], output.voxel_map);

		// groundseg
		groundSeg.enable =  read(tbl["ground"]["enable"], groundSeg.enable);
		groundSeg.distance =  read(tbl["ground"]["distance"], groundSeg.distance);
		groundSeg.angle =  read(tbl["ground"]["angle"], groundSeg.angle);
		groundSeg.planeDistance =  read(tbl["ground"]["planeDistance"], groundSeg.planeDistance);
		groundSeg.ground_min_z = read(tbl["ground"]["ground_min_z"], groundSeg.ground_min_z);
		groundSeg.ground_max_z = read(tbl["ground"]["ground_max_z"], groundSeg.ground_max_z);

		// filterNoise
		filterNoise.enable = read(tbl["filter"]["enable"], filterNoise.enable);
		filterNoise.filterMeanK = read(tbl["filter"]["filterMeanK"], filterNoise.filterMeanK);
		filterNoise.StddevMulThresh = read(tbl["filter"]["StddevMulThresh"], filterNoise.StddevMulThresh);

		// fdageParam
		fdageParam.replaceInputPly = read(tbl["fdage"]["replaceInputPly"], fdageParam.replaceInputPly);
		fdageParam.output_map = read(tbl["fdage"]["output_map"], fdageParam.output_map);
		fdageParam.select_m = read(tbl["fdage"]["select_m"], fdageParam.select_m);
		fdageParam.select_n = read(tbl["fdage"]["select_n"], fdageParam.select_n);
	}

	void save() const
	{
		// TODO: Implement
	}

 private:
	template <typename T>
	std::remove_cvref_t<T> read(toml::node_view<toml::node> node, T&& default_value)
	{
		// if (!node.is_value()) {
		// 	node.as_array()->push_back("MISSING");
		// 	missing_config = true;
		// 	std::cout << node << '\n';
		// 	return default_value;
		// }

		return node.value_or(default_value);
	}

 private:
	bool missing_config{false};
	bool wrong_config{false};
};

std::ostream& operator<<(std::ostream& out, Config const& config)
{
	out << "Config\n";

	out << "\tDataset\n";
	out << "\t\tFirst: " << config.dataset.first << '\n';
	out << "\t\tLast:  ";
	if (-1 == config.dataset.last) {
		out << -1 << '\n';
	} else {
		out << config.dataset.last << '\n';
	}
	out << "\t\tNum:   ";
	if (-1 == config.dataset.num) {
		out << -1 << '\n';
	} else {
		out << config.dataset.num << '\n';
	}

	out << "\tMap\n";
	out << "\t\tResolution: " << config.map.resolution << '\n';
	out << "\t\tLevels:     " << +config.map.levels << '\n';

	out << "\tIntegration\n";
	out << "\t\tDown sampling method:        "
	    << (ufo::DownSamplingMethod::NONE == config.integration.down_sampling_method
	            ? "none"
	            : (ufo::DownSamplingMethod::CENTER ==
	                       config.integration.down_sampling_method
	                   ? "center"
	                   : (ufo::DownSamplingMethod::CENTROID ==
	                              config.integration.down_sampling_method
	                          ? "centroid"
	                          : "uniform")))
	    << '\n';
	out << "\t\tHit depth:                   " << +config.integration.hit_depth << '\n';
	out << "\t\tMiss depth:                  " << +config.integration.miss_depth << '\n';
	out << "\t\tMin range:                   " << config.integration.min_range << '\n';
	out << "\t\tMax range:                   " << config.integration.max_range << '\n';
	out << "\t\tInflate unknown              " << config.integration.inflate_unknown
	    << '\n';
	out << "\t\tInflate unknown compensation "
	    << config.integration.inflate_unknown_compensation << '\n';
	out << "\t\tRay passthrough hits         " << config.integration.ray_passthrough_hits
	    << '\n';
	out << "\t\tInflate hits dist            " << config.integration.inflate_hits_dist
	    << '\n';
	out << "\t\tEarly stop distance:         " << config.integration.early_stop_distance
	    << '\n';
	out << "\t\tParallel:                    " << config.integration.parallel << '\n';
	out << "\t\tNum threads:                 " << config.integration.num_threads << '\n';
	out << "\t\tOnly valid:                  " << config.integration.only_valid << '\n';
	out << "\t\tSliding window size:         " << config.integration.sliding_window_size
	    << '\n';
	out << "\t\tPropagate:                   " << std::boolalpha << config.propagate
	    << '\n';

	out << "\tClustering\n";
	out << "\t\tCluster:      " << std::boolalpha << config.clustering.cluster << '\n';
	out << "\t\tMax distance: " << config.clustering.max_distance << '\n';
	out << "\t\tMin points:   " << config.clustering.min_points << '\n';
	out << "\t\tDepth:        " << +config.clustering.depth << '\n';

	out << "\tPrinting\n";
	out << "\t\tVerbose: " << std::boolalpha << config.printing.verbose << '\n';
	out << "\t\tDebug:   " << std::boolalpha << config.printing.debug << '\n';

	out << "\tOutput\n";
	out << "\t\tFilename:   " << config.output.filename << '\n';
	out << "\t\tHas color:  " << std::boolalpha << config.output.has_color << '\n';
	out << "\t\tRaycasting: " << config.output.raycasting << '\n';

	return out;
}

Config readConfig(std::filesystem::path path)
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

ufo::Color randomColor()
{
	static std::random_device                          rd;
	static std::mt19937                                gen(rd());
	static std::uniform_int_distribution<ufo::color_t> dis(0, -1);
	return {dis(gen), dis(gen), dis(gen)};
}

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

int run_pcd(int argc, char* argv[])
{
	if (1 >= argc) {
		std::cout << "[ERROR] Please running by: " << argv[0] << " [pcd_folder] [optional: config_file_path]";
		return 0;
	}

	std::filesystem::path path(argv[1]);
	std::string config_file_path;
	if (argc > 2)
		config_file_path = argv[2];
	else
		config_file_path = std::string(argv[1]) + "/dufomap.toml";

	auto config = readConfig(std::filesystem::path(config_file_path));
	std::cout << "[LOG] Step 1: Successfully read configuration from: " << config_file_path <<std::endl;
	ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION | ufo::MapType::LABEL> map(
	    config.map.resolution, config.map.levels);
	map.reserve(100'000'000);

	std::vector<std::filesystem::path> pcds;
	for (const auto& entry : std::filesystem::directory_iterator(path / "pcd")) {
		if (!entry.is_regular_file()) {
			continue;
		}
		std::size_t i = std::stoul(entry.path().stem());
		if (config.dataset.first <= i && config.dataset.last >= i) {
			pcds.push_back(entry.path().filename());
		}
	}
	std::ranges::sort(pcds);
	// std::cout << config << std::endl;
	pcds.resize(std::min(pcds.size(), config.dataset.num));

	ufo::Timing timing;
	timing.start("Total");

	ufo::PointCloudColor cloud_acc;

	std::cout << "[LOG] Step 2: Starting Processing data from: " << path << '\n';
	indicators::show_console_cursor(false);
	indicators::BlockProgressBar bar{
		indicators::option::BarWidth{50},
		indicators::option::Start{"["},
		indicators::option::End{"]"},
		indicators::option::PrefixText{"[LOG] Running dufomap "},
		indicators::option::ForegroundColor{indicators::Color::white},
		indicators::option::ShowElapsedTime{true},
		indicators::option::ShowRemainingTime{true},
		indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
	};
	
	for (std::size_t i{}; std::string filename : pcds) {
		// FIXME: looks like unicode::display_width will have some influence on the algorithm, haven't figure out yet.
		// bar.set_progress(100 * i / pcds.size());
		++i;
        if(i>1 && !config.printing.verbose){
            std::ostringstream log_msg;
            log_msg << "(" << i << "/" << pcds.size() << ") Processing: " << filename << " Time Cost: " 
                << config.integration.timing.lastSeconds() << "s";
            std::string spaces(10, ' ');
            log_msg << spaces;
            std::cout << "\r" <<log_msg.str() << std::flush;
        }

		ufo::PointCloudColor cloud;
		ufo::Pose6f          viewpoint;
		timing[1].start("Read");
		ufo::readPointCloudPCD(path / "pcd" / filename, cloud, viewpoint);
		timing[1].stop();

		cloud_acc.insert(std::end(cloud_acc), std::cbegin(cloud), std::cend(cloud));

		ufo::insertPointCloud(map, cloud, viewpoint.translation, config.integration,
		                      config.propagate);

		if (config.printing.verbose) {
			timing.setTag("Total " + std::to_string(i) + " of " + std::to_string(pcds.size()) +
						" (" + std::to_string(100 * i / pcds.size()) + "%)");
			timing[2] = config.integration.timing;
			timing.print(true, true, 2, 4);
		}
	}
	indicators::show_console_cursor(true);
	std::cout << "\033[0m\n[LOG] Step 3: Finished Processing data. Start saving map... " << std::endl;
	// bar.is_completed();
	if (!config.propagate) {
		timing[3].start("Propagate");
		map.propagateModified();
		timing[3].stop();
	}

	timing[4].start("Cluster");
	if (config.clustering.cluster) {
		cluster(map, config.clustering);
	}
	timing[4].stop();

	timing[5].start("Query");
	ufo::PointCloudColor cloud_static;
    ufo::PointCloudColor cloud_remove;

	if (config.output.voxel_map){
		// save voxel map: based on the config resolution
		for (auto node : map.query(
				ufo::pred::Leaf(0) && !ufo::pred::SeenFree() && ufo::pred::HitsMin(1))) {
			auto p = map.center(node);
			cloud_static.emplace_back(p, ufo::Color(255, 255, 255));
		}
		config.output.filename += "_voxel";
	}
	else{
		for (auto& p : cloud_acc)
			if (!map.seenFree(p))
            {
                cloud_static.push_back(p);
            }else{
                cloud_remove.push_back(p);
            }
	}
	timing[5].stop();

	timing[6].start("write");
	//ufo::writePointCloudPCD(path / (config.output.filename + ".pcd"), cloud_static);
	ufo::writePointCloudPLY(path / (config.output.filename + "_remove.ply"), cloud_remove);
	ufo::writePointCloudPLY_downsample(path / (config.output.filename + "downsample.ply"), cloud_static, 0.1, 0.1, 0.1);
	timing[6].stop();
	timing.stop();

	timing[2] = config.integration.timing;
	timing.print(true, true, 2, 4);
	std::cout << "[LOG]: Finished! ^v^.. Clean output map with " << cloud_static.size() << " points save in " << path / (config.output.filename + ".pcd") << '\n';
	return 0;
}

// 将旋转矩阵转换为四元数
void fromRotationMatrix_my(const float matrix[3][3], double& w, double& x, double& y, double& z) {
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

static std::vector<std::string> str_split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> res;
    if ("" == str)
    {
        return res;
    }

    for (size_t p = 0, q = 0; p != str.npos; p = q)
    {
        res.push_back(
            str.substr(p + (p != 0), (q = str.find(delim, p + 1)) - p - (p != 0)));
    }

    return res;
}

template <typename Map>
void retainMapRange(Map& map, size_t n, size_t m) {
    if (n > m || m > map.size()) return; // 确保范围有效
 
    auto begin_it = std::next(map.begin(), n - 1);
    auto end_it = std::next(map.begin(), m);
 
    map.erase(map.begin(), begin_it);
    map.erase(end_it, map.end());
}

void filterGroundPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr const& pc,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr& ground,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr& nonground,
	double groundFilterDistance,
	double groundFilterAngle) {
	if (pc->size() < 50) {
		cout << "Pointcloud in OctomapServer too small, skipping ground plane extraction";
		nonground = pc;
		return;
	}

	// plane detection for ground plane removal:
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

	// Create the segmentation object and set up:
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	seg.setOptimizeCoefficients(true);

	seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(groundFilterDistance);
	seg.setAxis(Eigen::Vector3f(0, 0, 1));
	seg.setEpsAngle(groundFilterAngle);

	// Create the filtering object
	seg.setInputCloud(pc);
	seg.segment(*inliers, *coefficients);
	if (inliers->indices.size() == 0) {
		cout << "PCL segmentation did not find any plane.";
		nonground = pc;
		return;
	}
	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	bool groundPlaneFound = false;
	extract.setInputCloud(pc);
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(*ground);
	if (inliers->indices.size() != pc->size()) {
		extract.setNegative(true);
		pcl::PointCloud<pcl::PointXYZRGB> cloud_out;
		extract.filter(cloud_out);
		*nonground += cloud_out;
	}
}

void filterGroundPlane_2(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr const& pc,
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& ground,
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& nonground,
	double groundFilterDistance,
	double groundFilterAngle) {
	if (pc->size() < 50) {
		cout << "Pointcloud in OctomapServer too small, skipping ground plane extraction";
		nonground = pc;
		return;
	}

	// plane detection for ground plane removal:
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

	// Create the segmentation object and set up:
	pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;
	seg.setOptimizeCoefficients(true);

	seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(groundFilterDistance);
	seg.setAxis(Eigen::Vector3f(0, 0, 1));
	seg.setEpsAngle(groundFilterAngle);

	// Create the filtering object
	seg.setInputCloud(pc);
	seg.segment(*inliers, *coefficients);
	if (inliers->indices.size() == 0) {
		cout << "PCL segmentation did not find any plane.";
		nonground = pc;
		return;
	}
	pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
	bool groundPlaneFound = false;
	extract.setInputCloud(pc);
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(*ground);
	if (inliers->indices.size() != pc->size()) {
		extract.setNegative(true);
		pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_out;
		extract.filter(cloud_out);


		*nonground += cloud_out;
	}
}

int run_ply_fdage(int argc, char* argv[])
{
	std::filesystem::path path(argv[1]);
	std::string config_file_path = argv[2];
	std::string vision_json_path = std::string(argv[1]) + "vision.json";
	std::string depthmap_dir = std::string(argv[1]) + "depthmap/";
	std::string ground_dir =  std::string(argv[1]) + "ground/";

	size_t select_m = 1;
	size_t select_n = 2548;

	ifstream ifs(vision_json_path, ios::in);
	auto j = nlohmann::json::parse(ifs);
	ifs.close();

	int location_num = j["sweepLocations"].size();
    std::map<std::string, SweepLocation>map_id_sweepLocation;

	for(int i = 0; i < location_num; i++){
		auto j_pose = j["sweepLocations"][i]["pose"];
		std::string uuid = j["sweepLocations"][i]["uuid"];

		std::vector<float> pose(7, 0.0);
		double t_x = j_pose["translation"]["x"];
		double t_y = j_pose["translation"]["y"];
		double t_z = j_pose["translation"]["z"];

		pose[0] = t_x;
		pose[1] = t_y;
		pose[2] = t_z;

		{
			double w = j_pose["rotation"]["w"];
			double x = j_pose["rotation"]["x"];
			double y = j_pose["rotation"]["y"];
			double z = j_pose["rotation"]["z"];
			// 将四元数转换为旋转矩阵
			float matrix[3][3];
			// 计算旋转矩阵的元素
			matrix[0][0] = 1 - 2 * (y * y + z * z);
			matrix[0][1] = 2 * (x * y - w * z);
			matrix[0][2] = 2 * (x * z + w * y);

			matrix[1][0] = 2 * (x * y + w * z);
			matrix[1][1] = 1 - 2 * (x * x + z * z);
			matrix[1][2] = 2 * (y * z - w * x);

			matrix[2][0] = 2 * (x * z - w * y);
			matrix[2][1] = 2 * (y * z + w * x);
			matrix[2][2] = 1 - 2 * (x * x + y * y);

			float matrix_rote_z[3][3] = { -1.0000000, -0.0000000,  0.0000000,
										0.0000000, -1.0000000,  0.0000000,
										0.0000000,  0.0000000,  1.0000000 };
			float result[3][3];//右乘以旋转矩阵matrix_rote_z
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					result[i][j] = 0.0f;
					for (int k = 0; k < 3; ++k) {
						result[i][j] += matrix[i][k] * matrix_rote_z[k][j];
					}
				}
			}

			//更新pose
			fromRotationMatrix_my(result, w, x, y, z);
			pose[3] = w;
			pose[4] = x;
			pose[5] = y;
			pose[6] = z;
	    }

		//insert map  xx_xx  make key
		int id0 = atoi(str_split(uuid, "_")[0].c_str());
		int id = atoi(str_split(uuid, "_")[1].c_str());

		std::ostringstream tmp_id0;
		tmp_id0  << std::setfill('0') << std::setw(6) << id0;
		std::ostringstream tmp_id;
		tmp_id  << std::setfill('0') << std::setw(6) << id;

		std::string str_id = tmp_id0.str() + tmp_id.str();

		SweepLocation ele;
		ele.uuid = uuid;
		ele.model_path = depthmap_dir + uuid + ".ply";
		ele.pose6f.x() = pose[0];
		ele.pose6f.y() = pose[1];
		ele.pose6f.z() = pose[2];

		ele.pose6f.qw() = pose[3];
		ele.pose6f.qx() = pose[4];
		ele.pose6f.qy() = pose[5];
		ele.pose6f.qz() = pose[6];

		if(std::filesystem::exists(ele.model_path)){
			map_id_sweepLocation.insert(std::make_pair(str_id, ele));
		}
		
    }

	//
	retainMapRange(map_id_sweepLocation, select_m, select_n);

	auto config = readConfig(std::filesystem::path(config_file_path));
	std::cout << "[LOG] Step 1: Successfully read configuration from: " << config_file_path <<std::endl;
	ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION | ufo::MapType::LABEL> map(
	    config.map.resolution, config.map.levels);
	map.reserve(100'000'000);


	Potree::AABB all_aabb;
	ufo::PointCloudColor cloud_acc;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_acc_ground(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::vector<size_t>cloud_acc_count;
	std::vector<size_t>cloud_acc_ground_count;

	int num_location = map_id_sweepLocation.size();
	int i = 0;
	ufo::Timing timing;
	timing.start("Total");

	std::random_device                          rd;
	std::mt19937                                gen(rd());
	std::uniform_int_distribution<> dis(0, 255);

	for(auto& pair:map_id_sweepLocation){
		SweepLocation ele = pair.second;
		std::string ply_path = ele.model_path;
		ufo::Pose6f viewpoint = ele.pose6f;

		i++;
		if(i>1 && !config.printing.verbose){
            std::ostringstream log_msg;
            log_msg << "(" << i << "/" << num_location << ") Processing: " << ele.uuid << " Time Cost: " 
                << config.integration.timing.lastSeconds() << "s";
            std::string spaces(10, ' ');
            log_msg << spaces;
            std::cout << "\r" <<log_msg.str() << std::flush;
        }

		// 随机着色
		ufo::Color color;
		color.red = dis(gen);
		color.green = dis(gen);
		color.blue = dis(gen);

		ufo::PointCloudColor cloud;		
		ufo::readPointCloudPLY(ply_path, cloud, all_aabb, color);
		//ufo::applyTransform(cloud, viewpoint);

		//原始点云建图
		ufo::PointCloudColor cloud_clone;	
		cloud_clone.resize(cloud.size());
		
		for (long i = 0; i < cloud.size(); i++){
		cloud_clone[i] = cloud[i];
		}
		ufo::applyTransform(cloud_clone, viewpoint);
		ufo::insertPointCloud(map, cloud_clone, viewpoint.translation, config.integration,
		                      config.propagate);

		/////////////////////////////////////
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		std::vector<float>vec_points(cloud.size()*3);
		for(int i = 0; i < cloud.size(); i++){
			vec_points[3*i + 0] = cloud[i].x;
			vec_points[3*i + 1] = cloud[i].y;
			vec_points[3*i + 2] = cloud[i].z;

			pcl::PointXYZRGB pt;
			pt.x = cloud[i].x;
			pt.y = cloud[i].y;
			pt.z = cloud[i].z;
			pt.r = color.red;
			pt.g = color.green;
			pt.b = color.blue;

			pcl_cloud->push_back(pt);
		}

		//Groud_seg groud_seg = Groud_seg();
		//std::string out_ply_path = std::string(argv[1]) + "ground_seg/" + ele.uuid + ".ply";
		//groud_seg.groud_seg_my(vec_points, out_ply_path);
		//groud_seg.groud_seg_patchworkpp(vec_points, out_ply_path);

		///// 仿照octomap 数据处理
		bool filterNoise = config.filterNoise.enable;  // 去噪filterNoise
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
		if (filterNoise) {
			pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor(true);
			sor.setInputCloud(pcl_cloud);
			sor.setMeanK(config.filterNoise.filterMeanK);
			sor.setStddevMulThresh(config.filterNoise.StddevMulThresh);
			sor.filter(*cloud_filtered);

			pcl::IndicesConstPtr remove_index = sor.getRemovedIndices();
			const int* idx_ptr = &(*remove_index)[0];
			size_t num = remove_index->size();
			
		}
		else {
			// remove NaN points
			std::vector<int> indices;
			pcl::removeNaNFromPointCloud(*pcl_cloud, *cloud_filtered, indices);
		}

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_nonground(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_ground(new pcl::PointCloud<pcl::PointXYZRGB>);

		// 过滤地面
		bool bfilterGroundPlane =  config.groundSeg.enable;
		if (bfilterGroundPlane) {
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cut_ground(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cut_nonground(new pcl::PointCloud<pcl::PointXYZRGB>);
			for(int i = 0; i < cloud_filtered->points.size(); i++){
				float z =  cloud_filtered->points[i].z;
				pcl::PointXYZRGB pt;
				pt.x = cloud_filtered->points[i].x;
				pt.y = cloud_filtered->points[i].y;
				pt.z = cloud_filtered->points[i].z;
				pt.r = color.red;
				pt.g = color.green;
				pt.b = color.blue;

				if(-1.7 < z && z < -0.5){
					cloud_cut_ground->push_back(pt);			
				}else{
					cloud_cut_nonground->push_back(pt);	
				}
			}

			double groundFilterDistance = config.groundSeg.distance;
			double groundFilterAngle = config.groundSeg.angle;
			// filterGroundPlane(cloud_filtered, pc_ground, pc_nonground, groundFilterDistance, groundFilterAngle);
			filterGroundPlane(cloud_cut_ground, pc_ground, pc_nonground, groundFilterDistance, groundFilterAngle);
			*pc_nonground +=  *cloud_cut_nonground;
			
			//save the pcd ground
			if (pc_ground->points.size() > 0) {
				// int img_index = atoi(str_split(ele.uuid, "_")[1].c_str());
				// std::ostringstream tmp_filename;
				// tmp_filename << ground_dir << std::setfill('0') <<
				// 	std::setw(6) << img_index << ".pcd"; std::string pcd_file =
				// 	tmp_filename.str(); pcl::io::savePCDFileBinary(pcd_file, *pc_ground);

				applyTransform(std::begin(pc_ground->points), std::end(pc_ground->points), viewpoint );
				cloud_acc_ground->insert(cloud_acc_ground->end(), pc_ground->begin(), pc_ground->end());
			}		
		}
		else{
			pc_nonground = cloud_filtered;
		}
		
		// 更新cloud
		size_t non_ground_points_num =  pc_nonground->points.size();
		cloud.clear();
		cloud.resize(non_ground_points_num);
		for(int i = 0; i < non_ground_points_num; i++){
			cloud[i].x = pc_nonground->points[i].x;
			cloud[i].y = pc_nonground->points[i].y;
			cloud[i].z = pc_nonground->points[i].z;

			cloud[i].red   =pc_nonground->points[i].r;
			cloud[i].green   =pc_nonground->points[i].g;
			cloud[i].blue   = pc_nonground->points[i].b;
		}

		cloud_acc_count.push_back(cloud.size());
		cloud_acc_ground_count.push_back(pc_ground->points.size());

		ufo::applyTransform(cloud, viewpoint);
		/////////
		/////////////////////////////////

		cloud_acc.insert(std::end(cloud_acc), std::cbegin(cloud), std::cend(cloud));

		// 去除噪声、地面点云后建图
		// ufo::insertPointCloud(map, cloud, viewpoint.translation, config.integration,
		//                       config.propagate);
		
		if (config.printing.verbose) {
			timing.setTag("Total " + std::to_string(i) + " of " + std::to_string(num_location) +
						" (" + std::to_string(100 * i / num_location) + "%)");
			timing[2] = config.integration.timing;
			timing.print(true, true, 2, 4);
		}
	}

	indicators::show_console_cursor(true);
	std::cout << "\033[0m\n[LOG] Step 3: Finished Processing data. Start saving map... " << std::endl;
	// bar.is_completed();
	if (!config.propagate) {
		timing[3].start("Propagate");
		map.propagateModified();
		timing[3].stop();
	}

	timing[4].start("Cluster");
	if (config.clustering.cluster) {
		cluster(map, config.clustering);
	}
	timing[4].stop();

	timing[5].start("Query");
	ufo::PointCloudColor cloud_static;
    ufo::PointCloudColor cloud_remove;

	if (config.output.voxel_map){
		// save voxel map: based on the config resolution
		for (auto node : map.query(
				ufo::pred::Leaf(0) && !ufo::pred::SeenFree() && ufo::pred::HitsMin(1))) {
			auto p = map.center(node);
			cloud_static.emplace_back(p, ufo::Color(255, 255, 255));
		}
		config.output.filename += "_voxel";
	}
	else{
		size_t count_index = 0; 
		size_t ply_index = 0;
		
		std::vector<size_t>cloud_static_count(cloud_acc_count.size()); // 保存每帧静态点云个数

		for (auto& p : cloud_acc)
		{
			if (!map.seenFree(p))
            {
                cloud_static.push_back(p);

				cloud_static_count[ply_index]++;
            }else{
                cloud_remove.push_back(p);
            }

			count_index++;
			if(count_index == cloud_acc_count[ply_index]){ // 达到下一个
				count_index = 0;
				ply_index++;
			}
		}
		
		// 将地面点云加上
		count_index = 0; 
		ply_index = 0;
		for(auto &pt : cloud_acc_ground->points ){
			cloud_static_count[ply_index]++;

			auto p = cloud_static[0];
			p.x = pt.x;
			p.y = pt.y;
			p.z = pt.z;
			p.red = 255;
			p.green = 255;
			p.blue = 0;
			cloud_static.push_back(p);

			count_index++;

			if(count_index == cloud_acc_ground_count[ply_index]){ // 达到下一个
				count_index = 0;
				ply_index++;
			}		
		}

		count_index = 0; 
		ply_index = 0;
		ufo::PointCloudColor save_cloud;
		save_cloud.resize(cloud_static_count[ply_index]);
		for (auto& p : cloud_static){
			save_cloud[count_index] = p;
			count_index++;

			if(count_index == cloud_static_count[ply_index]){ // 达到下一个

				auto it = map_id_sweepLocation.begin();
				std::advance(it, ply_index); 
    			
				SweepLocation ele = it->second;
				std::string ply_path = ele.model_path;
							
				{	// 转回原始ply坐标系中	
					ufo::Pose6f viewpoint = ele.pose6f;
					double w =viewpoint.qw();
					double x = viewpoint.qx();
					double y = viewpoint.qy();
					double z = viewpoint.qz();
					
					auto r = viewpoint.rotation.rotMatrix();

					Eigen::MatrixXd pose_4_4(4, 4);
					pose_4_4 << r[0], r[1], r[2], viewpoint.x(),
												r[3], r[4], r[5], viewpoint.y(),
												r[6], r[7], r[8], viewpoint.z(),
												0,      0,       0,        1;

					// 计算矩阵的逆
					Eigen::MatrixXd inverseMatrix = pose_4_4.inverse();
					
					//更新pose
					float result[3][3];
					result[0][0] = inverseMatrix(0, 0);  result[0][1] = inverseMatrix(0, 1);  result[0][2] = inverseMatrix(0, 2);
					result[1][0] = inverseMatrix(1, 0);  result[1][1] = inverseMatrix(1, 1);  result[1][2] = inverseMatrix(1, 2);
					result[2][0] = inverseMatrix(2, 0);  result[2][1] = inverseMatrix(2, 1);  result[2][2] = inverseMatrix(2, 2);
					fromRotationMatrix_my(result, w, x, y, z);
					
					viewpoint.qw() = w;
					viewpoint.qx() = x;
					viewpoint.qy() = y;
					viewpoint.qz() = z;

					viewpoint.x() = inverseMatrix(0, 3);
					viewpoint.y() =  inverseMatrix(1, 3);
					viewpoint.z() =  inverseMatrix(2, 3);  			

					ufo::applyTransform(save_cloud, viewpoint);
				}
			
				std::cout << ply_path << std::endl;
				//ufo::writePointCloudPLY(ply_path, save_cloud);

				count_index = 0;
				ply_index++;
				if(ply_index < cloud_static_count.size()){
					save_cloud.resize(cloud_static_count[ply_index]);
				}
				
			}
			
		}
	}
	
	timing[5].stop();

	timing[6].start("write");
	//ufo::writePointCloudPCD(path / (config.output.filename + ".pcd"), cloud_static);
	//ufo::writePointCloudPLY(path / (config.output.filename + "_remove.ply"), cloud_remove);
	ufo::Color color_remove;
	color_remove.red = 255;
	color_remove.green = 0;
	color_remove.blue = 0;
	std::string save_remove_name = "_remove_" + std::to_string(select_m) + "_" + std::to_string(select_n) + ".ply";
	std::string save_static_name = "_downsample_" + std::to_string(select_m) + "_" + std::to_string(select_n) + ".ply";
	writePointCloudPLY_downsample_remove(path / (config.output.filename + save_remove_name ), cloud_remove, 0.1, 0.1, 0.1, color_remove);
	ufo::writePointCloudPLY_downsample(path / (config.output.filename + save_static_name), cloud_static, 0.1, 0.1, 0.1);
	timing[6].stop();
	timing.stop();

	timing[2] = config.integration.timing;
	timing.print(true, true, 2, 4);
	std::cout << "[LOG]: Finished! ^v^.. Clean output map with " << cloud_static.size() << " points save in " << path / (config.output.filename + ".pcd") << '\n';
	return 0;
}


int run_ply_fdage_2(int argc, char* argv[])
{
	std::filesystem::path path(argv[1]);
	std::string config_file_path = argv[2];
	std::string vision_json_path = std::string(argv[1]) + "vision.json";
	std::string depthmap_dir = std::string(argv[1]) + "depthmap/";
	std::string ground_dir =  std::string(argv[1]) + "ground/";

	ifstream ifs(vision_json_path, ios::in);
	auto j = nlohmann::json::parse(ifs);
	ifs.close();

	int location_num = j["sweepLocations"].size();
    std::map<std::string, SweepLocation>map_id_sweepLocation;

	for(int i = 0; i < location_num; i++){
		auto j_pose = j["sweepLocations"][i]["pose"];
		std::string uuid = j["sweepLocations"][i]["uuid"];

		std::vector<float> pose(7, 0.0);
		double t_x = j_pose["translation"]["x"];
		double t_y = j_pose["translation"]["y"];
		double t_z = j_pose["translation"]["z"];

		pose[0] = t_x;
		pose[1] = t_y;
		pose[2] = t_z;

		{
			double w = j_pose["rotation"]["w"];
			double x = j_pose["rotation"]["x"];
			double y = j_pose["rotation"]["y"];
			double z = j_pose["rotation"]["z"];
			// 将四元数转换为旋转矩阵
			float matrix[3][3];
			// 计算旋转矩阵的元素
			matrix[0][0] = 1 - 2 * (y * y + z * z);
			matrix[0][1] = 2 * (x * y - w * z);
			matrix[0][2] = 2 * (x * z + w * y);

			matrix[1][0] = 2 * (x * y + w * z);
			matrix[1][1] = 1 - 2 * (x * x + z * z);
			matrix[1][2] = 2 * (y * z - w * x);

			matrix[2][0] = 2 * (x * z - w * y);
			matrix[2][1] = 2 * (y * z + w * x);
			matrix[2][2] = 1 - 2 * (x * x + y * y);

			float matrix_rote_z[3][3] = { -1.0000000, -0.0000000,  0.0000000,
										0.0000000, -1.0000000,  0.0000000,
										0.0000000,  0.0000000,  1.0000000 };
			float result[3][3];//右乘以旋转矩阵matrix_rote_z
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					result[i][j] = 0.0f;
					for (int k = 0; k < 3; ++k) {
						result[i][j] += matrix[i][k] * matrix_rote_z[k][j];
					}
				}
			}

			//更新pose
			fromRotationMatrix_my(result, w, x, y, z);
			pose[3] = w;
			pose[4] = x;
			pose[5] = y;
			pose[6] = z;
	    }

		//insert map  xx_xx  make key
		int id0 = atoi(str_split(uuid, "_")[0].c_str());
		int id = atoi(str_split(uuid, "_")[1].c_str());

		std::ostringstream tmp_id0;
		tmp_id0  << std::setfill('0') << std::setw(6) << id0;
		std::ostringstream tmp_id;
		tmp_id  << std::setfill('0') << std::setw(6) << id;

		std::string str_id = tmp_id0.str() + tmp_id.str();

		SweepLocation ele;
		ele.uuid = uuid;
		ele.model_path = depthmap_dir + uuid + ".ply";
		ele.pose6f.x() = pose[0];
		ele.pose6f.y() = pose[1];
		ele.pose6f.z() = pose[2];

		ele.pose6f.qw() = pose[3];
		ele.pose6f.qx() = pose[4];
		ele.pose6f.qy() = pose[5];
		ele.pose6f.qz() = pose[6];

		if(std::filesystem::exists(ele.model_path)){
			map_id_sweepLocation.insert(std::make_pair(str_id, ele));
		}
		
    }

	auto config = readConfig(std::filesystem::path(config_file_path));
	std::cout << "[LOG] Step 1: Successfully read configuration from: " << config_file_path <<std::endl;
	ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION | ufo::MapType::LABEL> map(
	    config.map.resolution, config.map.levels);
	map.reserve(100'000'000);

	// retainMapRange 从1开始
	size_t select_m = 0;
	size_t select_n = map_id_sweepLocation.size() - 1;

	if(config.fdageParam.select_m >= 0 && config.fdageParam.select_n >= 0 
	&& config.fdageParam.select_m <= config.fdageParam.select_n){
		select_m = config.fdageParam.select_m;
		select_n = config.fdageParam.select_n;
	}
	retainMapRange(map_id_sweepLocation, select_m+1, select_n+1);

	Potree::AABB all_aabb;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr all_pc_ground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	int num_location = map_id_sweepLocation.size();
	int i = 0;
	ufo::Timing timing;
	timing.start("Total");

	std::random_device                          rd;
	std::mt19937                                gen(rd());
	std::uniform_int_distribution<> dis(0, 255);

	for(auto& pair:map_id_sweepLocation){
		SweepLocation ele = pair.second;
		std::string ply_path = ele.model_path;
		ufo::Pose6f viewpoint = ele.pose6f;

		i++;
		if(i>1 && !config.printing.verbose){
            std::ostringstream log_msg;
            log_msg << "(" << i << "/" << num_location << ") Processing: " << ele.uuid << " Time Cost: " 
                << config.integration.timing.lastSeconds() << "s";
            std::string spaces(10, ' ');
            log_msg << spaces;
            std::cout << "\r" <<log_msg.str() << std::flush;
        }

		// 随机着色
		ufo::Color color;
		color.red = dis(gen);
		color.green = dis(gen);
		color.blue = dis(gen);

		ufo::PointCloudColor cloud;		
		Potree::AABB aabb;
		ufo::readPointCloudPLY(ply_path, cloud, aabb, color);
		////////////////////////////
		// pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc_nonground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		// pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc_ground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	  
		// // 过滤地面
		// bool bfilterGroundPlane =  config.groundSeg.enable;
		// if (bfilterGroundPlane) {
		// 	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_cut_ground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		// 	for(int i = 0; i < cloud.size(); i++){
		// 		float z =  cloud[i].z;
		// 		pcl::PointXYZRGBNormal pt;
		// 		pt.x =  cloud[i].x;
		// 		pt.y =  cloud[i].y;
		// 		pt.z =  cloud[i].z;
		// 		pt.r = color.red;
		// 		pt.g = color.green;
		// 		pt.b = color.blue;
			
		// 		if(config.groundSeg.ground_min_z < z && z < config.groundSeg.ground_max_z){
		// 			cloud_cut_ground->push_back(pt);			
		// 		}
		// 	}

		// 	double groundFilterDistance = config.groundSeg.distance;
		// 	double groundFilterAngle = config.groundSeg.angle;
		// 	filterGroundPlane_2(cloud_cut_ground, pc_ground, pc_nonground, groundFilterDistance, groundFilterAngle);
		// 	ufo::applyTransform(std::begin(*pc_ground), std::end(*pc_ground), viewpoint);

		// 	*all_pc_ground +=  *pc_ground; 		
		// }
		///////////////////////////
		ufo::applyTransform(cloud, viewpoint);
		all_aabb.update(aabb);

		ufo::insertPointCloud(map, cloud, viewpoint.translation, config.integration,
		                      config.propagate);
		
		if (config.printing.verbose) {
			timing.setTag("Total " + std::to_string(i) + " of " + std::to_string(num_location) +
						" (" + std::to_string(100 * i / num_location) + "%)");
			timing[2] = config.integration.timing;
			timing.print(true, true, 2, 4);
		}
	}

	// query
	ufo::PointCloudColor cloud_static;
    ufo::PointCloudColor cloud_remove;

	for(auto& pair:map_id_sweepLocation){
		SweepLocation ele = pair.second;
		std::string ply_path = ele.model_path;
		ufo::Pose6f viewpoint = ele.pose6f;

		// 随机着色
		ufo::Color color;
		color.red = dis(gen);
		color.green = dis(gen);
		color.blue = dis(gen);

		Potree::PlyPointReader *reader = new Potree::PlyPointReader(ply_path);
		size_t numPoints = reader->numPoints();
		std::vector<Potree::Point> vec_points;
		Potree::AABB aabb = reader->getAABB(vec_points);

		////
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc_nonground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc_ground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	   //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr static_cut_nonground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		// 过滤地面
		bool bfilterGroundPlane =  config.groundSeg.enable;
		if (bfilterGroundPlane) {
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_cut_ground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_cut_nonground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			for(int i = 0; i < vec_points.size(); i++){
				float z =  vec_points[i].position.z;
				pcl::PointXYZRGBNormal pt;
				pt.x = vec_points[i].position.x;
				pt.y = vec_points[i].position.y;
				pt.z = vec_points[i].position.z;
				pt.r = color.red;
				pt.g = color.green;
				pt.b = color.blue;
				pt.normal_x = vec_points[i].normal.x;
				pt.normal_y = vec_points[i].normal.y;
				pt.normal_z = vec_points[i].normal.z;

				if(config.groundSeg.ground_min_z < z && z < config.groundSeg.ground_max_z){
					cloud_cut_ground->push_back(pt);			
				}else{
					cloud_cut_nonground->push_back(pt);	
					//static_cut_nonground->push_back(pt); // 范围外保留
				}
			}

			double groundFilterDistance = config.groundSeg.distance;
			double groundFilterAngle = config.groundSeg.angle;
			// filterGroundPlane(cloud_filtered, pc_ground, pc_nonground, groundFilterDistance, groundFilterAngle);
			filterGroundPlane_2(cloud_cut_ground, pc_ground, pc_nonground, groundFilterDistance, groundFilterAngle);
			*pc_nonground +=  *cloud_cut_nonground;  // pc_nonground 会放去图中判断是否应该去除
			
			//save the pcd ground
			if (pc_ground->points.size() > 0) {
				// int img_index = atoi(str_split(ele.uuid, "_")[1].c_str());
				// std::ostringstream tmp_filename;
				// tmp_filename << ground_dir << std::setfill('0') <<
				// 	std::setw(6) << img_index << ".pcd"; std::string pcd_file =
				// 	tmp_filename.str(); pcl::io::savePCDFileBinary(pcd_file, *pc_ground);
		
			}		
		}
		else{
			for(int i = 0; i < vec_points.size(); i++){
				float z =  vec_points[i].position.z;
				pcl::PointXYZRGBNormal pt;
				pt.x = vec_points[i].position.x;
				pt.y = vec_points[i].position.y;
				pt.z = vec_points[i].position.z;
				pt.r = color.red;
				pt.g = color.green;
				pt.b = color.blue;
				pt.normal_x = vec_points[i].normal.x;
				pt.normal_y = vec_points[i].normal.y;
				pt.normal_z = vec_points[i].normal.z;

				pc_nonground->push_back(pt);
			}
		}

		////

		auto r = viewpoint.rotation.rotMatrix();
		auto t =  viewpoint.translation;
		ufo::PointCloudColor pt;
		pt.resize(1);
		
		std::vector<Potree::Point> vec_points_static;
		std::vector<Potree::Point> vec_points_remove;

		//
		for(auto &p :  pc_ground->points){
			double x = p.x;
			double y = p.y;
			double z = p.z;

			Potree::Point point;
			point.position.x = x;
			point.position.y = y;
			point.position.z = z;

			point.normal.x = p.normal_x;
			point.normal.y = p.normal_y;
			point.normal.z = p.normal_z;

			point.color.x = 255;
			point.color.y = 255;
			point.color.z = 0;

			vec_points_static.push_back(point);	

			if(config.fdageParam.output_map){
				pt[0].x   = r[0] * x + r[1] * y + r[2] * z + t.x;
				pt[0].y   = r[3] * x + r[4] * y + r[5] * z + t.y;
				pt[0].z    = r[6] * x + r[7] * y + r[8] * z + t.z;

				pt[0].red = 255;
				pt[0].green = 255;
				pt[0].blue = 0;
				cloud_static.push_back(pt[0]);
				}
		}

		// 地面附近外的点保留
		// for(auto &p :  static_cut_nonground->points){
		// 	double x = p.x;
		// 	double y = p.y;
		// 	double z = p.z;

		// 	Potree::Point point;
		// 	point.position.x = x;
		// 	point.position.y = y;
		// 	point.position.z = z;

		// 	point.normal.x = p.normal_x;
		// 	point.normal.y = p.normal_y;
		// 	point.normal.z = p.normal_z;

		// 	point.color.x = p.r;
		// 	point.color.y = p.g;
		// 	point.color.z = p.b;

		// 	vec_points_static.push_back(point);	

		// 	if(config.fdageParam.output_map){
		// 		pt[0].x   = r[0] * x + r[1] * y + r[2] * z + t.x;
		// 		pt[0].y   = r[3] * x + r[4] * y + r[5] * z + t.y;
		// 		pt[0].z    = r[6] * x + r[7] * y + r[8] * z + t.z;

		// 		pt[0].red = p.r;
		// 		pt[0].green = p.g;
		// 		pt[0].blue = p.b;
		// 		cloud_static.push_back(pt[0]);
		// 		}
		// }

		for(auto &p :  pc_nonground->points){
			double x = p.x;
			double y = p.y;
			double z = p.z;

			pt[0].x   = r[0] * x + r[1] * y + r[2] * z + t.x;
		    pt[0].y   = r[3] * x + r[4] * y + r[5] * z + t.y;
		    pt[0].z    = r[6] * x + r[7] * y + r[8] * z + t.z;

			pt[0].red = p.r;
			pt[0].green = p.g;
			pt[0].blue = p.b;

			Potree::Point point;
			point.position.x = x;
			point.position.y = y;
			point.position.z = z;

			point.normal.x = p.normal_x;
			point.normal.y = p.normal_y;
			point.normal.z = p.normal_z;

			point.color.x = p.r;
			point.color.y = p.g;
			point.color.z = p.b;

			if (!map.seenFree(pt[0]))
			{
				vec_points_static.push_back(point);		
				if(config.fdageParam.output_map){
					cloud_static.push_back(pt[0]);
				}
				
			}else{
				point.color.x = 255;
				point.color.y = 0;
				point.color.z = 0;
				vec_points_remove.push_back(point);

				if(config.fdageParam.output_map){
					cloud_remove.push_back(pt[0]);
				}
			}
		}
		
		reader->close();
		delete reader;

		if( config.fdageParam.replaceInputPly)
		{
				std::cout << "replaceInputPly: "<< ply_path << std::endl;
				ufo::writePointCloudPLY_bin(ply_path, vec_points_static);
		}
		
	}

	indicators::show_console_cursor(true);
	std::cout << "\033[0m\n[LOG] Step 3: Finished Processing data. Start saving map... " << std::endl;
	// bar.is_completed();
	if (!config.propagate) {
		timing[3].start("Propagate");
		map.propagateModified();
		timing[3].stop();
	}

	timing[4].start("Cluster");
	if (config.clustering.cluster) {
		cluster(map, config.clustering);
	}
	timing[4].stop();

	if(config.fdageParam.output_map)
	{
		std::cout << "output map begin" << std::endl;
		ufo::Color color_remove;
		color_remove.red = 255;
		color_remove.green = 0;
		color_remove.blue = 0;
		std::string save_remove_name = "_remove_" + std::to_string(select_m) + "_" + std::to_string(select_n) + ".ply";
		std::string save_static_name = "_downsample_" + std::to_string(select_m) + "_" + std::to_string(select_n) + ".ply";
		writePointCloudPLY_downsample_remove(path / (config.output.filename + save_remove_name ), cloud_remove, 0.1, 0.1, 0.1, color_remove);
		ufo::writePointCloudPLY_downsample(path / (config.output.filename + save_static_name), cloud_static, 0.1, 0.1, 0.1);
		std::cout << "output map end" << std::endl;
	}
	
	return 0;
}


int run_fdage_laz(int argc, char* argv[])
{
	std::filesystem::path path(argv[1]);
	std::string config_file_path = argv[2];
	std::string vision_json_path = std::string(argv[1]) + "capture/slam_data_new.json";
	std::string las_dir = std::string(argv[1]) + "capture/";
	std::string ply_dir = std::string(argv[1]) + "capture/";
	std::string ground_dir =  std::string(argv[1]) + "ground/";
	std::string new_laz_dir = std::string(argv[1]) + "new_laz/";

	ifstream ifs(vision_json_path, ios::in);
	auto j = nlohmann::json::parse(ifs);
	ifs.close();

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
			//ele.model_path = las_dir + std::to_string(floor_index) + "_" + uuid + ".ply";
			ele.model_path = las_dir + std::to_string(floor_id) + "_" + uuid + "_depth.laz";
			
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
	

	auto config = readConfig(std::filesystem::path(config_file_path));
	std::cout << "[LOG] Step 1: Successfully read configuration from: " << config_file_path <<std::endl;
	ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION | ufo::MapType::LABEL> map(
	    config.map.resolution, config.map.levels);
	map.reserve(100'000'000);

	// retainMapRange 从1开始
	size_t select_m = 0;
	size_t select_n = map_id_sweepLocation.size() - 1;

	if(config.fdageParam.select_m >= 0 && config.fdageParam.select_n >= 0 
	&& config.fdageParam.select_m <= config.fdageParam.select_n){
		select_m = config.fdageParam.select_m;
		select_n = config.fdageParam.select_n;
	}
	retainMapRange(map_id_sweepLocation, select_m+1, select_n+1);

	Potree::AABB all_aabb;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr all_pc_ground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	int num_location = map_id_sweepLocation.size();
	int i = 0;
	ufo::Timing timing;
	timing.start("Total");

	std::random_device                          rd;
	std::mt19937                                gen(rd());
	std::uniform_int_distribution<> dis(0, 255);

	for(auto& pair:map_id_sweepLocation){
		SweepLocation ele = pair.second;
		std::string las_path = ele.model_path;
		ufo::Pose6f viewpoint = ele.pose6f;

		i++;
		if(i>1 && !config.printing.verbose){
            std::ostringstream log_msg;
            log_msg << "(" << i << "/" << num_location << ") Processing: " << ele.uuid << " Time Cost: " 
                << config.integration.timing.lastSeconds() << "s";
            std::string spaces(10, ' ');
            log_msg << spaces;
            std::cout << "\r" <<log_msg.str() << std::flush;
        }

		// 随机着色
		ufo::Color color;
		color.red = dis(gen);
		color.green = dis(gen);
		color.blue = dis(gen);

		ufo::PointCloudColor cloud;		
		Potree::AABB aabb;
		// ufo::readPointCloudPLY(ply_path, cloud, aabb, color);
		ufo::readPointCloudLAZ(las_path, cloud, aabb, color);
		
		ufo::applyTransform(cloud, viewpoint);
		all_aabb.update(aabb);

		ufo::insertPointCloud(map, cloud, viewpoint.translation, config.integration,
		                      config.propagate);
		
		if (config.printing.verbose) {
			timing.setTag("Total " + std::to_string(i) + " of " + std::to_string(num_location) +
						" (" + std::to_string(100 * i / num_location) + "%)");
			timing[2] = config.integration.timing;
			timing.print(true, true, 2, 4);
		}
	}

	// query
	ufo::PointCloudColor cloud_static;
    ufo::PointCloudColor cloud_remove;

	for(auto& pair:map_id_sweepLocation){
		SweepLocation ele = pair.second;
		std::string las_path = ele.model_path;
		ufo::Pose6f viewpoint = ele.pose6f;

		// 随机着色
		ufo::Color color;
		color.red = dis(gen);
		color.green = dis(gen);
		color.blue = dis(gen);

		// Potree::PlyPointReader *reader = new Potree::PlyPointReader(ply_path);
		// size_t numPoints = reader->numPoints();
		// std::vector<Potree::Point> vec_points;
		// Potree::AABB aabb = reader->getAABB(vec_points);

		ufo::PointCloudColor cloud;		
		Potree::AABB aabb;
		ufo::readPointCloudLAZ(las_path, cloud, aabb, color);
		size_t numPoints = cloud.size();

		std::vector<Potree::Point> vec_points(numPoints);

            for(int i = 0; i < numPoints; i++){
                Potree::Point point;
                point.position.x = cloud[i].x;
                point.position.y = cloud[i].y;
                point.position.z = cloud[i].z;

                point.color.x = cloud[i].red;
                point.color.y = cloud[i].green;
                point.color.z = cloud[i].blue;

                vec_points[i] = point;
            }


		////
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc_nonground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc_ground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	   //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr static_cut_nonground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		// 过滤地面
		bool bfilterGroundPlane =  config.groundSeg.enable;
		bfilterGroundPlane = false;
		if (bfilterGroundPlane) {
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_cut_ground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_cut_nonground(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			for(int i = 0; i < vec_points.size(); i++){
				float z =  vec_points[i].position.z;
				pcl::PointXYZRGBNormal pt;
				pt.x = vec_points[i].position.x;
				pt.y = vec_points[i].position.y;
				pt.z = vec_points[i].position.z;
				pt.r = color.red;
				pt.g = color.green;
				pt.b = color.blue;
				pt.normal_x = vec_points[i].normal.x;
				pt.normal_y = vec_points[i].normal.y;
				pt.normal_z = vec_points[i].normal.z;

				if(config.groundSeg.ground_min_z < z && z < config.groundSeg.ground_max_z){
					cloud_cut_ground->push_back(pt);			
				}else{
					cloud_cut_nonground->push_back(pt);	
					//static_cut_nonground->push_back(pt); // 范围外保留
				}
			}

			double groundFilterDistance = config.groundSeg.distance;
			double groundFilterAngle = config.groundSeg.angle;
			// filterGroundPlane(cloud_filtered, pc_ground, pc_nonground, groundFilterDistance, groundFilterAngle);
			filterGroundPlane_2(cloud_cut_ground, pc_ground, pc_nonground, groundFilterDistance, groundFilterAngle);
			*pc_nonground +=  *cloud_cut_nonground;  // pc_nonground 会放去图中判断是否应该去除
			
			//save the pcd ground
			if (pc_ground->points.size() > 0) {
				// int img_index = atoi(str_split(ele.uuid, "_")[1].c_str());
				// std::ostringstream tmp_filename;
				// tmp_filename << ground_dir << std::setfill('0') <<
				// 	std::setw(6) << img_index << ".pcd"; std::string pcd_file =
				// 	tmp_filename.str(); pcl::io::savePCDFileBinary(pcd_file, *pc_ground);
		
			}		
		}
		else{
			for(int i = 0; i < vec_points.size(); i++){
				float z =  vec_points[i].position.z;
				pcl::PointXYZRGBNormal pt;
				pt.x = vec_points[i].position.x;
				pt.y = vec_points[i].position.y;
				pt.z = vec_points[i].position.z;
				pt.r = color.red;
				pt.g = color.green;
				pt.b = color.blue;
				pt.normal_x = vec_points[i].normal.x;
				pt.normal_y = vec_points[i].normal.y;
				pt.normal_z = vec_points[i].normal.z;

				pc_nonground->push_back(pt);
			}
		}

		////

		auto r = viewpoint.rotation.rotMatrix();
		auto t =  viewpoint.translation;
		ufo::PointCloudColor pt;
		pt.resize(1);
		
		std::vector<Potree::Point> vec_points_static;
		std::vector<Potree::Point> vec_points_remove;
		Potree::AABB static_point_aabb;

		//
		for(auto &p :  pc_ground->points){
			double x = p.x;
			double y = p.y;
			double z = p.z;

			Potree::Point point;
			point.position.x = x;
			point.position.y = y;
			point.position.z = z;

			point.normal.x = p.normal_x;
			point.normal.y = p.normal_y;
			point.normal.z = p.normal_z;

			point.color.x = 255;
			point.color.y = 255;
			point.color.z = 0;

			vec_points_static.push_back(point);	


			if(config.fdageParam.output_map){
				pt[0].x   = r[0] * x + r[1] * y + r[2] * z + t.x;
				pt[0].y   = r[3] * x + r[4] * y + r[5] * z + t.y;
				pt[0].z    = r[6] * x + r[7] * y + r[8] * z + t.z;

				pt[0].red = 255;
				pt[0].green = 255;
				pt[0].blue = 0;
				cloud_static.push_back(pt[0]);

                Potree::Vector3<double> pt_tmp(pt[0].x, pt[0].y, pt[0].z);
                static_point_aabb.update(pt_tmp);
			}
		}

		for(auto &p :  pc_nonground->points){
			double x = p.x;
			double y = p.y;
			double z = p.z;

			pt[0].x   = r[0] * x + r[1] * y + r[2] * z + t.x;
		    pt[0].y   = r[3] * x + r[4] * y + r[5] * z + t.y;
		    pt[0].z    = r[6] * x + r[7] * y + r[8] * z + t.z;

			pt[0].red = p.r;
			pt[0].green = p.g;
			pt[0].blue = p.b;

			Potree::Point point;
			point.position.x = x;
			point.position.y = y;
			point.position.z = z;

			point.normal.x = p.normal_x;
			point.normal.y = p.normal_y;
			point.normal.z = p.normal_z;

			point.color.x = p.r;
			point.color.y = p.g;
			point.color.z = p.b;

			if (!map.seenFree(pt[0]))
			{
				vec_points_static.push_back(point);		

				Potree::Vector3<double> pt_tmp(pt[0].x, pt[0].y, pt[0].z);
                static_point_aabb.update(pt_tmp);

				if(config.fdageParam.output_map){
					cloud_static.push_back(pt[0]);
				}
				
			}else{
				point.color.x = 255;
				point.color.y = 0;
				point.color.z = 0;
				vec_points_remove.push_back(point);

				if(config.fdageParam.output_map){
					cloud_remove.push_back(pt[0]);
				}
			}
		}

		if( config.fdageParam.replaceInputPly)
		{
			    int  floor_id = ele.floor_id;
				string ply_path = ply_dir + std::to_string(floor_id) + "_" + ele.uuid + ".ply";
				
				ufo::writePointCloudPLY_bin(ply_path, vec_points_static);

                string las_path = new_laz_dir + std::to_string(floor_id) + "_" + ele.uuid + ".laz";
				std::cout << "replaceInputFile: "<< las_path << std::endl;
                ufo::writePointCloudLAS(las_path, vec_points_static, static_point_aabb);
		}
		
	}

	indicators::show_console_cursor(true);
	std::cout << "\033[0m\n[LOG] Step 3: Finished Processing data. Start saving map... " << std::endl;
	// bar.is_completed();
	if (!config.propagate) {
		timing[3].start("Propagate");
		map.propagateModified();
		timing[3].stop();
	}

	timing[4].start("Cluster");
	if (config.clustering.cluster) {
		cluster(map, config.clustering);
	}
	timing[4].stop();

	if(config.fdageParam.output_map)
	{
		std::cout << "output map begin" << std::endl;
		ufo::Color color_remove;
		color_remove.red = 255;
		color_remove.green = 0;
		color_remove.blue = 0;
		std::string save_remove_name = "_remove_" + std::to_string(select_m) + "_" + std::to_string(select_n) + ".ply";
		std::string save_static_name = "_downsample_" + std::to_string(select_m) + "_" + std::to_string(select_n) + ".ply";
		writePointCloudPLY_downsample_remove(path / (config.output.filename + save_remove_name ), cloud_remove, 0.1, 0.1, 0.1, color_remove);
		ufo::writePointCloudPLY_downsample(path / (config.output.filename + save_static_name), cloud_static, 0.1, 0.1, 0.1);
		std::cout << "output map end" << std::endl;
	}
	
	return 0;
}
int main(int argc, char* argv[])
{
	 //run_pcd(argc, argv);
	 //run_ply_fdage(argc, argv);
	 //run_ply_fdage_2(argc, argv);
	 run_fdage_laz(argc, argv);
	 return 0;
}

