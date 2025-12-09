#pragma once

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
	double		   forward_hit_extend_distance = 1.8;
	int kMinHitFrames = -1;          
	int kMinLifeSpanFrames = -1;
	int dp = 1; // dynamic voxel neighbor distance
	double void_known_ratio_min = 0.2;
	bool enable_seenfree_debug = false;
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
	float ground_min_y=-1.7;
	float ground_max_y=-0.5;
	float ground_max_y_height=-1.0;
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
		map.forward_hit_extend_distance = read(tbl["important"]["forward_hit_extend_distance"], map.forward_hit_extend_distance);
		map.kMinHitFrames = read(tbl["important"]["kMinHitFrames"], map.kMinHitFrames);
		map.kMinLifeSpanFrames = read(tbl["important"]["kMinLifeSpanFrames"], map.kMinLifeSpanFrames);
		map.dp = read(tbl["important"]["dp"], map.dp);
		map.void_known_ratio_min = read(tbl["important"]["void_known_ratio_min"], map.void_known_ratio_min);
		map.enable_seenfree_debug = read(tbl["important"]["enable_seenfree_debug"], map.enable_seenfree_debug);

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
		groundSeg.ground_min_y = read(tbl["ground"]["ground_min_y"], groundSeg.ground_min_y);
		groundSeg.ground_max_y = read(tbl["ground"]["ground_max_y"], groundSeg.ground_max_y);
		groundSeg.ground_max_y_height = read(tbl["ground"]["ground_max_y_height"], groundSeg.ground_max_y_height);

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
