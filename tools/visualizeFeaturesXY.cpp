/**
 * @file visualizeFeaturesXY.cpp
 * @brief visualizeFeaturesXY
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Features/VisualizeFeatures.h"
#include "VideoParsing/IO/FeatureIO.h"
#include "VideoParsing/Utils/ScopedTimer.h"

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
      ("viz_mode,v", po::value<int>()->default_value(1), "Visualization 0:OFF 1:Standard 2:Shifted")
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

  if(!bfs::exists(fp)) {
    std::cout << fp << " does not exist\n";
    return EXIT_FAILURE;
  }
  if(!bfs::is_regular_file(fp)) {
    std::cout << fp << " is not a regular file\n";
    return EXIT_FAILURE;
  }


  Eigen::Matrix<float, 2, Eigen::Dynamic> featuresXY;
  vp::FeatureInfo finfo;
  {
    vp::ScopedTimer tmr("Loading Feature Matrix from " + fp.string() + " ...... ");
    std::tie(featuresXY, finfo) = vp::readFeatures(fp.string());
  }

  std::string dataset_config_file = finfo.dataset_name + ".cfg";
  if(vm.count("dataset"))
    dataset_config_file = vm.at("dataset").as<std::string>();

  std::cout << "Dataset Config= " << dataset_config_file << "\n";


  vp::Dataset dataset(dataset_config_file);

  const int viz_mode = vm.at("viz_mode").as<int>();

  std::cout << "Dataset = " << dataset.name() << "\n";
  std::cout << "Frames = [" << finfo.start_frame << "," << finfo.start_frame + finfo.number_of_frames - 1 << "]\n";
  std::cout << "viz_mode = " << viz_mode << "\n";


  if(viz_mode == 1)
    vp::visualizeFeaturesXY(featuresXY, dataset, finfo.start_frame, finfo.number_of_frames, fp.filename().string());
  else if(viz_mode == 2)
    vp::visualizeFeaturesXYwShifting(featuresXY, dataset, finfo.start_frame, finfo.number_of_frames, fp.filename().string());

  return EXIT_SUCCESS;
}
