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

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

#include "laswriter.hpp"
#include "lasreader.hpp"

#include "laswriter.hpp"
#include "Point.h"
#include "AABB.h"
#include "json.hpp"

using namespace std;

void read_las(string las_path, vector<float>& vec_points) {
	LASreadOpener lasReadOpener;
	lasReadOpener.set_file_name(las_path.c_str());

	LASreader* lasReader = lasReadOpener.open();

	LASheader header = lasReader->header;

	double x_scale = header.x_scale_factor;
	double y_scale = header.y_scale_factor;
	double z_scale = header.z_scale_factor;

	double x_offset = header.x_offset;
	double y_offset = header.y_offset;
	double z_offset = header.z_offset;

	unsigned int point_num = header.number_of_point_records;
	vector<float> vec_points_tmp(point_num*3);
    int i = 0;
	while (lasReader->read_point()) {
		const double px = lasReader->point.get_X() * header.x_scale_factor + header.x_offset;
		const double py = lasReader->point.get_Y() * header.y_scale_factor + header.y_offset;
		const double pz = lasReader->point.get_Z() * header.z_scale_factor + header.z_offset;

		vec_points_tmp[3*i + 0] = px;
		vec_points_tmp[3*i + 1] = py;
		vec_points_tmp[3*i + 2] = pz;
        i++;
	}
	vec_points = vec_points_tmp;
	lasReader->close();
}

struct SweepLocation {
	std::string uuid;
	ufo::Pose6f pose6f;
	std::string ply_path;
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

int run_laz_data(int argc, char* argv[])
{
	std::filesystem::path path(argv[1]);
	std::string config_file_path = argv[2];
	std::string vision_json_path = std::string(argv[1]) + "capture/slam_data_new.json";
	std::string laz_dir = std::string(argv[1]) + "capture/";
    std::string ply_dir = std::string(argv[1]) + "ply/";
    std::string new_laz_dir = std::string(argv[1]) + "new_laz/";

	
	ifstream ifs(vision_json_path, ios::in);
	auto j = nlohmann::json::parse(ifs);
	ifs.close();

	auto config = readConfig(std::filesystem::path(config_file_path));
	std::cout << "[LOG] Step 1: Successfully read configuration from: " << config_file_path <<std::endl;
	ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION | ufo::MapType::LABEL> map(
	    config.map.resolution, config.map.levels);
	map.reserve(100'000'000);

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
	
    for (int floor_index = 0; floor_index < j["views_info"].size(); floor_index++) {
	    auto& j_floor_ele = j["views_info"][floor_index];

	    int floor_id = j_floor_ele["id_floor"];
        int poses_num = j_floor_ele["list_pose"].size();

        // select [m, n] to process
        size_t select_m = 0;
        size_t select_n = j_floor_ele["list_pose"].size() - 1;

        if(config.fdageParam.select_m >= 0 && config.fdageParam.select_n >= 0 
        && config.fdageParam.select_m <= config.fdageParam.select_n){
            select_m = config.fdageParam.select_m;

            if(config.fdageParam.select_n < select_n){
                select_n = config.fdageParam.select_n;
            }
        }

        poses_num = select_n;

        for (int pose_index = select_m; pose_index <= select_n; pose_index++) 
        {
              //cout << pose_index << " / " << poses_num << endl;
            
             if(pose_index>0 && !config.printing.verbose){
                std::ostringstream log_msg;
                std::string filename =  std::to_string(floor_index) + "_" + std::to_string(pose_index) + "_depth.laz";

                log_msg << "(" << pose_index << "/" <<poses_num << ") Processing: " << filename << " Time Cost: " 
                    << config.integration.timing.lastSeconds() << "s";
                std::string spaces(10, ' ');

                log_msg << spaces;
                std::cout << "\r" <<log_msg.str() << std::flush;
             }

            auto& j_pose_list = j_floor_ele["list_pose"][pose_index];
            double w = j_pose_list[0];
            double x = j_pose_list[1];
            double y = j_pose_list[2];
            double z = j_pose_list[3];

            double tvec_x = j_pose_list[4];
            double tvec_y = j_pose_list[5];
            double tvec_z = j_pose_list[6];

             ufo::Pose6f          viewpoint;
            viewpoint.x() = tvec_x;
		    viewpoint.y() = tvec_y;
		    viewpoint.z() = tvec_z;

		    viewpoint.qw() = w;
		    viewpoint.qx() = x;
		    viewpoint.qy() = y;
		    viewpoint.qz() = z;

            ufo::PointCloudColor cloud;		
		    Potree::AABB aabb;
            // 随机着色
		    ufo::Color color;
		    color.red = 255;
		    color.green = 0;
		    color.blue = 0;

            string las_path = laz_dir + std::to_string(floor_id) + "_" + std::to_string(pose_index) + "_depth.laz";
		    ufo::readPointCloudLAZ(las_path, cloud, aabb, color);
            
            // 随机着色
            std::random_device                          rd;
	        std::mt19937                                gen(rd());
	        std::uniform_int_distribution<> dis(0, 255);
            for(auto &p : cloud) {
	                p.red   = dis(gen);
	             	p.green   = dis(gen);
	             	p.blue   = dis(gen);
            }

            ufo::applyTransform(cloud, viewpoint);
            ufo::insertPointCloud(map, cloud, viewpoint.translation, config.integration,
		                      config.propagate);

            if (config.printing.verbose) {
                timing.setTag("Total " + std::to_string(pose_index) + " of " + std::to_string(poses_num) +
                            " (" + std::to_string(100 * pose_index / poses_num) + "%)");
                timing[2] = config.integration.timing;
                timing.print(true, true, 2, 4);
            }

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

    // query
	ufo::PointCloudColor cloud_static;
    ufo::PointCloudColor cloud_remove;

    for (int floor_index = 0; floor_index < j["views_info"].size(); floor_index++) {
	    auto& j_floor_ele = j["views_info"][floor_index];

	    int floor_id = j_floor_ele["id_floor"];

        // select [m, n] to process
        size_t select_m = 0;
        size_t select_n = j_floor_ele["list_pose"].size() - 1;

        if(config.fdageParam.select_m >= 0 && config.fdageParam.select_n >= 0 
        && config.fdageParam.select_m <= config.fdageParam.select_n){
            select_m = config.fdageParam.select_m;

            if(config.fdageParam.select_n < select_n){
                select_n = config.fdageParam.select_n;
            }
        }

        for (int pose_index = select_m; pose_index <= select_n; pose_index++) {
            auto& j_pose_list = j_floor_ele["list_pose"][pose_index];
            double w = j_pose_list[0];
            double x = j_pose_list[1];
            double y = j_pose_list[2];
            double z = j_pose_list[3];

            double tvec_x = j_pose_list[4];
            double tvec_y = j_pose_list[5];
            double tvec_z = j_pose_list[6];

            ufo::Pose6f          viewpoint;
            viewpoint.x() = tvec_x;
		    viewpoint.y() = tvec_y;
		    viewpoint.z() = tvec_z;

		    viewpoint.qw() = w;
		    viewpoint.qx() = x;
		    viewpoint.qy() = y;
		    viewpoint.qz() = z;

            ufo::PointCloudColor cloud;		
		    Potree::AABB aabb;
         
		    ufo::Color color;
		    color.red = 255;
		    color.green = 0;
		    color.blue = 0;

            string las_path = laz_dir + std::to_string(floor_id) + "_" + std::to_string(pose_index) + "_depth.laz";
		    ufo::readPointCloudLAZ(las_path, cloud, aabb, color);

            // 随机着色
            std::random_device                          rd;
	        std::mt19937                                gen(rd());
	        std::uniform_int_distribution<> dis(0, 255);
            for(auto &p : cloud) {
	                p.red   = dis(gen);
	             	p.green   = dis(gen);
	             	p.blue   = dis(gen);
            }

            int points_num = cloud.size();
            std::vector<Potree::Point> vec_points_src(points_num);

            for(int i = 0; i < points_num; i++){
                Potree::Point point;
                point.position.x = cloud[i].x;
                point.position.y = cloud[i].y;
                point.position.z = cloud[i].z;

                point.color.x = cloud[i].red;
                point.color.y = cloud[i].green;
                point.color.z = cloud[i].blue;

                vec_points_src[i] = point;
            }


            ufo::applyTransform(cloud, viewpoint);
		    
            std::vector<Potree::Point> vec_points_static;
            std::vector<Potree::Point> vec_points_remove;
            Potree::AABB static_point_aabb;

            for(int i = 0; i < points_num; i++)
            {
                auto p = cloud[i];
                if (!map.seenFree(p))
                {
                    double p_x = vec_points_src[i].position.x;
                    double p_y = vec_points_src[i].position.y;
                    double p_z = vec_points_src[i].position.z;

                    Potree::Vector3<double> point(p_x, p_y, p_z);
                    static_point_aabb.update(point);

                    vec_points_static.push_back(vec_points_src[i]);		
                        if(config.fdageParam.output_map){
                            cloud_static.push_back(p);
                        }
                }else{
                    vec_points_src[i].color.x = 255;
                    vec_points_src[i].color.y = 0;
                    vec_points_src[i].color.z = 0;
                    vec_points_remove.push_back(vec_points_src[i]);

                    p.red = 255;
                    p.green = 0;
                    p.blue = 0;

                    std::cout << "fsdfsgg" << std::endl;

                    if(config.fdageParam.output_map){
                        cloud_remove.push_back(p);
                    }
			    }
            }

            if( config.fdageParam.replaceInputPly)
		    {
                 string ply_path = ply_dir + std::to_string(floor_id) + "_" + std::to_string(pose_index) + ".ply";

				std::cout << "replaceInputPly: "<< ply_path << std::endl;
				ufo::writePointCloudPLY_bin(ply_path, vec_points_static);

                string las_path = new_laz_dir + std::to_string(floor_id) + "_" + std::to_string(pose_index) + ".laz";
                ufo::writePointCloudLAS(las_path, vec_points_static, static_point_aabb);
		    }
                     
        }

    }

	timing[5].stop();

	timing[6].start("write");
	ufo::writePointCloudPLY(path / (config.output.filename + "_remove.ply"), cloud_remove);
	ufo::writePointCloudPLY_downsample(path / (config.output.filename + "downsample.ply"), cloud_static, 0.04, 0.04, 0.04);
	timing[6].stop();
	timing.stop();

	timing[2] = config.integration.timing;
	timing.print(true, true, 2, 4);
	//std::cout << "[LOG]: Finished! ^v^.. Clean output map with " << cloud_static.size() << " points save in " << path / (config.output.filename + ".pcd") << '\n';
    std::cout << "[LOG]: Finished! ^v^.. Clean output map with " << cloud_static.size() << " points save" << '\n';
	return 0;
}

int main(int argc, char* argv[])
{
	 run_laz_data(argc, argv);
	 return 0;
}
