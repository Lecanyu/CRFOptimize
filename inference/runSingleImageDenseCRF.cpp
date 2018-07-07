///**
// * @file runSingleImageDenseCRF.cpp
// * @brief run Single Image Dense CRF with command line options
// *
// * @author Abhijit Kundu
// */
//
//#include "VideoParsing/IO/Dataset.h"
//#include "VideoParsing/Utils/SemanticLabel.h"
//#include "VideoParsing/Utils/ProgramOptionsHelper.h"
//#include "VideoParsing/DenseGM/Dense2DGridGM.h"
//#include <opencv2/highgui/highgui.hpp>
//#include <boost/format.hpp>
//#include <iostream>
//
//
//int main(int argc, char **argv) {
//  namespace po = boost::program_options;
//  using std::cout;
//
//  po::options_description all_options;
//
//  {
//    po::options_description config_options("DenseCRF");
//    config_options.add_options()
//          ("help,h", "Help screen")
//          ("dataset,d", po::value<std::string>()->default_value("Camvid05VD.cfg"), "Dataset Config File")
//          ("frame_id,f",  po::value<int>()->required(), "FrameId")
//          ("iterations,i",  po::value<int>()->default_value(5), "Number of iterations")
//          ("print_config,p",  po::value<bool>()->default_value(false), "Print Configuration")
//          ("out_dir",  po::value<std::string>()->default_value("."), "Output directory")
//          ;
//    all_options.add(config_options);
//  }
//
//  all_options.add(vp::makeKernelParamsOptions());
//
//  po::variables_map vm;
//
//  try {
//    po::store(parse_command_line(argc, argv, all_options), vm);
//    po::notify(vm);
//    vp::verfyKernelParamsOptions(vm);
//  } catch (const po::error &ex) {
//    std::cerr << ex.what() << '\n';
//
//    cout << all_options << '\n';
//    return EXIT_FAILURE;
//  }
//
//  if (vm.count("help")) {
//    cout << all_options << '\n';
//    return EXIT_SUCCESS;
//  }
//
//
//  const int frame_id = vm["frame_id"].as<int>();
//  const int iterations = vm["iterations"].as<int>();
//  const vp::Dataset dataset(vm["dataset"].as<std::string>());
//
//
//  typedef std::vector<float> StdVectorf;
//  const StdVectorf& XYpottswt = vm.at("XY.PottsWt").as<StdVectorf>();
//  const StdVectorf& XYsigmas = vm.at("XY.sigmas").as<StdVectorf>();
//
//  const StdVectorf& XYRGBpottswt = vm.at("XYRGB.PottsWt").as<StdVectorf>();
//  const StdVectorf& XYRGBsigmas = vm.at("XYRGB.sigmas").as<StdVectorf>();
//
//  const std::string save_filename = (boost::format("%s/%s_%06d.png")
//      % vm["out_dir"].as<std::string>() % dataset.name() % frame_id).str();
//
//
//  if (vm["print_config"].as<bool>()) {
//    std::cout << "---------- Config -----------\n";
//    vp::printPOVariablesMap(vm);
//    std::cout << "-----------------------------\n";
//  }
//
//  // Load the color image
//  cv::Mat img = dataset.loadImage(frame_id);
//  if (img.empty()) {
//    cout << "Failed to load Color image!\n";
//    return EXIT_FAILURE;
//  }
//  const int W = img.cols;  // width of the grid
//  const int H = img.rows;  // height of the grid
//
//  // Setup the CRF model
//  vp::Dense2DGridGM crf(W, H, dataset.loadUnaryAsNegLogProb(frame_id));
//
//  crf.addPairwiseGaussian(XYsigmas[0], XYsigmas[1], XYpottswt, vp::NORMALIZE_SYMMETRIC);
//
//  crf.addPairwiseBilateral(XYRGBsigmas[0], XYRGBsigmas[1], XYRGBsigmas[2], XYRGBsigmas[3],
//                           XYRGBsigmas[4], img, XYRGBpottswt, vp::NORMALIZE_SYMMETRIC);
//
//
//  // Do map inference
//  Eigen::VectorXi map = crf.computeMAPestimate(iterations);
//
//
//  cv::imwrite(save_filename, vp::getRGBImageFromLabels(map, W, H, crf.numberOfLabels()));
//
//  return EXIT_SUCCESS;
//}
