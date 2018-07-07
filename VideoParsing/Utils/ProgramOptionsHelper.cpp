/**
 * @file ProgramOptionsHelper.cpp
 * @brief ProgramOptionsHelper
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Utils/ProgramOptionsHelper.h"
#include <iostream>

namespace vp {

boost::program_options::options_description makeDatasetConfigOptions() {
  namespace po = boost::program_options;

  po::options_description dataset_config_options("Dataset Config Options");
  dataset_config_options.add_options()
      ("dataset,d", po::value<std::string>()->default_value("camvid-05VD-640x480.cfg"), "Dataset Config File")
      ("start_frame,s",  po::value<int>()->required(), "start_frame")
      ("number_of_frames,n",  po::value<int>()->required(), "number_of_frames")
      ;

  return dataset_config_options;
}

boost::program_options::options_description makeKernelParamsOptions() {
  namespace po = boost::program_options;
  typedef std::vector<float> StdVectorf;

  po::options_description kernel_options("Kernel Options");
  kernel_options.add_options()
      ("XY.PottsWt",  po::value<StdVectorf>()->multitoken()->default_value({0}, "0"), "XY kernel Potts Weight")
      ("XY.sigmas",  po::value<StdVectorf>()->multitoken()->default_value({3, 3}, "3 3"), "XY kernel sigmas")

      ("XYRGB.PottsWt",  po::value<StdVectorf>()->multitoken()->default_value({0}, "0"), "XYRGB kernel Potts Weight")
      ("XYRGB.sigmas", po::value<StdVectorf>()->multitoken()->default_value({25, 25, 5, 5, 5}, "25 25 5 5 5"), "XYRGB kernel sigmas")

      ("XYRGBUV.PottsWt",  po::value<StdVectorf>()->multitoken()->default_value({0}, "0"), "XYRGBUV kernel Potts Weight")
      ("XYRGBUV.sigmas", po::value<StdVectorf>()->multitoken()->default_value({25, 25, 5, 5, 5, 1.2f, 1.2f}, "25 25 5 5 5 1.2 1.2"), "XYRGB kernel sigmas")

      ("XYT.PottsWt",  po::value<StdVectorf>()->multitoken()->default_value({0}, "0"), "XYT kernel Potts Weight")
      ("XYT.sigmas", po::value<StdVectorf>()->multitoken()->default_value({0.6f, 0.6f, 4}, "0.6, 0.6 4"), "XYT kernel sigmas")

      ("XYTRGB.PottsWt",  po::value<StdVectorf>()->multitoken()->default_value({0}, "0"), "XYTRGB kernel Potts Weight")
      ("XYTRGB.sigmas", po::value<StdVectorf>()->multitoken()->default_value({20, 20, 20, 5, 5, 5}, "20 20 20 5 5 5"), "XYTRGB kernel sigmas")

      ("XYTRGBUV.PottsWt",  po::value<StdVectorf>()->multitoken()->default_value({0}, "0"), "XYTRGBUV kernel Potts Weight")
      ("XYTRGBUV.sigmas", po::value<StdVectorf>()->multitoken()->default_value({20, 20, 20, 5, 5, 5, 4, 4}, "20 20 20 5 5 5 4 4"), "XYTRGBUV kernel sigmas")
      ;

  return kernel_options;
}

void verfyKernelParamsOptions(const boost::program_options::variables_map& vm) {
  namespace po = boost::program_options;
  typedef std::vector<float> StdVectorf;

  if (!vm["XY.sigmas"].empty() && (vm["XY.sigmas"].as<StdVectorf >()).size() != 2)
    throw po::error("Option XY.sigmas has invalid number of entries");

  if (!vm["XYRGB.sigmas"].empty() && (vm["XYRGB.sigmas"].as<StdVectorf >()).size() != 5)
    throw po::error("Option XYRGB.sigmas has invalid number of entries");

  if (!vm["XYRGBUV.sigmas"].empty() && (vm["XYRGBUV.sigmas"].as<StdVectorf >()).size() != 7)
    throw po::error("Option XYRGBUV.sigmas has invalid number of entries");

  if (!vm["XYT.sigmas"].empty() && (vm["XYT.sigmas"].as<StdVectorf >()).size() != 3)
    throw po::error("Option XYTRGB.sigmas has invalid number of entries");

  if (!vm["XYTRGB.sigmas"].empty() && (vm["XYTRGB.sigmas"].as<StdVectorf >()).size() != 6)
    throw po::error("Option XYTRGB.sigmas has invalid number of entries");

  if (!vm["XYTRGBUV.sigmas"].empty() && (vm["XYTRGBUV.sigmas"].as<StdVectorf >()).size() != 8)
    throw po::error("Option XYTRGBUV.sigmas has invalid number of entries");

}

boost::program_options::options_description makeHOSegementPotentialOptions() {
  namespace po = boost::program_options;

  po::options_description hoseg_options("HigherOrder Segment Potential Options");
  hoseg_options.add_options()
     ("HOSeg.param1",  po::value<float>()->default_value(0), "HO Seg Potential param1 (e.g 0.0025)")
     ("HOSeg.param2",  po::value<float>()->default_value(1.0f), "HO Seg Potential param2")
     ;

  return hoseg_options;
}

boost::program_options::options_description makeHOTrackPotentialOptions() {
  namespace po = boost::program_options;

  po::options_description hotrack_options("HigherOrder Track Potential Options");
  hotrack_options.add_options()
     ("HOTrack.all_same",  po::value<float>()->default_value(0.0f), "HO Track cost for all same labels")
     ("HOTrack.all_not_same",  po::value<float>(), "HO Track cost for all not same")
     ;

  return hotrack_options;
}

void printPOVariablesMap(const boost::program_options::variables_map& vm) {
  typedef std::vector<float> StdVectorf;

  for (const auto& it : vm) {
    std::cout << it.first.c_str() << " = ";
    auto& value = it.second.value();
    if (auto v = boost::any_cast<int>(&value))
      std::cout << *v;
    else if (auto v = boost::any_cast<float>(&value))
      std::cout << *v;
    else if (auto v = boost::any_cast<double>(&value))
      std::cout << *v;
    else if (auto v = boost::any_cast<std::string>(&value))
      std::cout << *v;
    else if (auto v = boost::any_cast<StdVectorf>(&value)) {
      for (auto v_val : *v)
        std::cout << v_val << " ";
    }
    else if (auto v = boost::any_cast<std::vector<std::string>>(&value)) {
      for (auto v_val : *v)
        std::cout << v_val << "; ";
    }
    else
      std::cout << "UnknownType";
    std::cout << "\n";
  }
}

}  // namespace vp


