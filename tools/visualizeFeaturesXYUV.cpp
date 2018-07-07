/*
 * visualizeXYUVFeatures.cpp
 *
 *  Created on: Nov 5, 2015
 *      Author: vvineet1
 */

#include "VideoParsing/Features/VisualizeFeatures.h"
#include "VideoParsing/IO/FeatureIO.h"
#include "VideoParsing/Utils/ScopedTimer.h"
#include "VideoParsing/Utils/ColorizeFlow.h"
#include "VideoParsing/Core/EigenOpenCVUtils.h"
#include "VideoParsing/Utils/VisualizationHelper.h"
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>


int main(int argc, char **argv) {

  namespace po = boost::program_options;

  po::options_description generic_options("Generic Options");
  generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
  config_options.add_options()
      ("dataset,d", po::value<std::string>(), "Dataset config file")
      ("feature_file,f", po::value<std::string>()->required(), "Feature File path")
      ;

  po::options_description cmdline_options;
  cmdline_options.add(generic_options).add(config_options);

  po::variables_map vm;

  try {
    po::store(parse_command_line(argc, argv, cmdline_options), vm);
    po::notify(vm);
  } catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  if (vm.count("help")) {
    std::cout << cmdline_options << '\n';
    return EXIT_SUCCESS;
  }

  namespace bfs = boost::filesystem;
  bfs::path fp(vm.at("feature_file").as<std::string>());

  Eigen::Matrix<float, 2, Eigen::Dynamic> featuresXY;
  Eigen::Matrix<float, 2, Eigen::Dynamic> featuresUV;
  vp::FeatureInfo finfo;
  {
    vp::ScopedTimer tmr("Loading Feature Matrix from " + fp.string() + " ......\n");

    Eigen::Matrix<float, 4, Eigen::Dynamic> featuresXYUV;

    std::tie(featuresXYUV, finfo) = vp::readFeatures(fp.string());
    std::cout << "Feat Dimension = " << finfo.dimension << "\n";

    std::cout << "MaxCoeff = " << featuresXYUV.rowwise().maxCoeff().transpose() << "\n";
    std::cout << "MinCoeff = " << featuresXYUV.rowwise().minCoeff().transpose() << "\n";

    featuresXY = featuresXYUV.topRows<2>();
    featuresUV = featuresXYUV.bottomRows<2>();

    std::cout << "Feature Loading  ";
  }

  std::string dataset_config_file = finfo.dataset_name + ".cfg";
  if(vm.count("dataset"))
    dataset_config_file = vm.at("dataset").as<std::string>();

  std::cout << "Dataset Config= " << dataset_config_file << "\n";


  vp::Dataset dataset(dataset_config_file);

  std::cout << "Dataset = " << dataset.name() << "\n";
  std::cout << "Frames = [" << finfo.start_frame << "," << finfo.start_frame + finfo.number_of_frames - 1 << "]\n";


  vp::visualizeFeaturesXY(featuresXY, dataset, finfo.start_frame, finfo.number_of_frames, fp.filename().string());

  {
    std::vector<cv::Mat> depth_maps(finfo.number_of_frames);
    const int WH = finfo.width * finfo.height;
    for (int t = 0; t < finfo.number_of_frames; ++t) {

      Eigen::Matrix2Xf Xflow = featuresUV.block(0, t * WH, 2,  WH);
      cv::Mat flow = convertToImage32F(Xflow.transpose(), finfo.width, finfo.height);
      depth_maps[t] = vp::colorizeFlowWithMagnitude(flow);
    }

    vp::visualizeImages(depth_maps, finfo.start_frame, "Optimized Flows", 0);
  }


  return EXIT_SUCCESS;
}

