/**
 * @file joinFeatures.cpp
 * @brief code to join two feature files
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/FeatureIO.h"
#include "VideoParsing/Utils/ScopedTimer.h"
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <iostream>


bool checkCompatbility(const vp::FeatureInfo& fiA, const vp::FeatureInfo& fiB) {
  return (fiA.dataset_name == fiB.dataset_name) && (fiA.start_frame == fiB.start_frame)
      && (fiA.number_of_frames == fiB.number_of_frames) && (fiA.height == fiB.height)
      && (fiA.width == fiB.width) && (fiA.scalar_size == fiB.scalar_size);
}


int main(int argc, char **argv) {

  if (argc < 4) {
    std::cout << "ERROR: parsing inputs\n";
    std::cout << "Usage: " << argv[0] << " feature_fileA feature_fileB out_feat_name\n";
    return EXIT_FAILURE;
  }

  namespace bfs = boost::filesystem;
  bfs::path fpA(argv[1]);
  if(!bfs::exists(fpA)) {
    std::cout << fpA << " does not exist\n";
    return EXIT_FAILURE;
  }
  if(!bfs::is_regular_file(fpA)) {
    std::cout << fpA << " is not a regular file\n";
    return EXIT_FAILURE;
  }

  bfs::path fpB(argv[2]);
  if(!bfs::exists(fpB)) {
    std::cout << fpB << " does not exist\n";
    return EXIT_FAILURE;
  }
  if(!bfs::is_regular_file(fpB)) {
    std::cout << fpB << " is not a regular file\n";
    return EXIT_FAILURE;
  }



  Eigen::Matrix<float, 2, Eigen::Dynamic> featuresA;
  vp::FeatureInfo finfoA;
  {
    vp::ScopedTimer tmr("Loading Feature Matrix from " + fpA.string() + " ...... ");
    std::tie(featuresA, finfoA) = vp::readFeatures(fpA.string());
  }

  Eigen::Matrix<float, 2, Eigen::Dynamic> featuresB;
  vp::FeatureInfo finfoB;
  {
    vp::ScopedTimer tmr("Loading Feature Matrix from " + fpB.string() + " ...... ");
    std::tie(featuresB, finfoB) = vp::readFeatures(fpB.string());
  }

  if(!checkCompatbility(finfoA, finfoB)) {
    std::cout << "Incompatible Features\n";
    return EXIT_FAILURE;
  }

  if (featuresA.size() != featuresB.size()) {
    std::cout << "Incompatible Feature Sizes\n";
    return EXIT_FAILURE;
  }

  std::string dataset_config_file = finfoA.dataset_name + ".cfg";
  std::cout << "Dataset Config= " << dataset_config_file << "\n";

  vp::Dataset dataset(dataset_config_file);


  Eigen::Matrix<float, 4, Eigen::Dynamic> features(4, featuresA.cols());
  features.topRows<2>() = featuresA;
  features.bottomRows<2>() = featuresB;

  std::cout << "Saving as " << argv[3] << std::endl;
  vp::saveFeatures(features, dataset, finfoA.start_frame, argv[3]);

  return EXIT_SUCCESS;
}
