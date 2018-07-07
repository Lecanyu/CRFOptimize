///**
// * @file runVideoGMwFeatures.cpp
// * @brief runVideoGMwFeatures
// *
// * @author Abhijit Kundu
// */
//
//#include "VideoParsing/DenseGM/VideoGM.h"
//#include "VideoParsing/IO/Dataset.h"
//#include "VideoParsing/IO/FeatureIO.h"
//#include "VideoParsing/Utils/SemanticLabel.h"
//#include "VideoParsing/Utils/FilterHelper.h"
//#include "VideoParsing/Utils/ScopedTimer.h"
//#include "VideoParsing/Utils/VisualizationHelper.h"
//#include "VideoParsing/Utils/ProgramOptionsHelper.h"
//
//#include <boost/assert.hpp>
//#include <iostream>
//
//vp::VideoGM setupGM(const boost::program_options::variables_map& vm) {
//
//  vp::Dataset dataset(vm.at("dataset").as<std::string>());
//  dataset.print();
//
//  const int start_frame = vm.at("start_frame").as<int>();
//  const int number_of_frames = vm.at("number_of_frames").as<int>();
//  const std::string feat_filepath = vm.at("feature_file").as<std::string>();
//
//  std::vector<cv::Mat> images;
//  std::vector<Eigen::MatrixXf> unaries;
//  {
//    vp::ScopedTimer tmr("Loading Images and Unaries ..");
//
//    images = dataset.loadImageSequence(start_frame, number_of_frames);
//    unaries = dataset.loadUnariesAsProb(start_frame, number_of_frames);
//
//#pragma omp parallel for
//    for (int i = 0; i < number_of_frames; ++i) {
//      Eigen::MatrixXf& unary = unaries[i];
//
//      // This makes sure unary prob are not zero
//      unary.array() += 0.0001f;
//      unary = unary.array().rowwise() / unary.colwise().sum().eval().array();
//
//      // Take negative log prob (This is mandatory)
//      unary = -unary.array().log();
//    }
//  }
//
//  // Some checking
//  BOOST_VERIFY(images.size() == unaries.size());
//  BOOST_VERIFY(unaries.size() == std::size_t(number_of_frames));
//
//  // Width Height Channels
//  const int W = images.front().cols;  // width of the grid
//  const int H = images.front().rows;  // height of the grid
//  const int M = unaries.front().rows();
//  const int WH = W * H;
//
//  for (int i = 0; i < number_of_frames; ++i) {
//    const cv::Mat& image = images[i];
//    BOOST_VERIFY(image.cols == W);
//    BOOST_VERIFY(image.rows == H);
//
//    BOOST_VERIFY(unaries[i].cols() == WH);
//  }
//
//  vp::VideoGM vgm(W, H, number_of_frames, M);
//
//  {
//    vp::ScopedTimer tmr("Adding Unary factors to CRF..");
//    vgm.addUnaries(unaries);
//  }
//
//  typedef std::vector<float> StdVectorf;
//
//  // Add spatial only factors
//  {
//    const float pottswt = vm.at("XY.PottsWt").as<StdVectorf>().at(0);
//    if(pottswt > 0) {
//      vp::ScopedTimer tmr("Adding 2D XY pairwise terms .. ");
//      const StdVectorf& sigmas = vm.at("XY.sigmas").as<StdVectorf>();
//      vgm.addSpatialPairwiseGaussian(sigmas.at(0), sigmas.at(1),
//                                     vp::PottsFunction(pottswt),
//                                     vp::NORMALIZE_SYMMETRIC);
//    }
//  }
//
//
//  {
//    const float pottswt = vm.at("XYRGB.PottsWt").as<StdVectorf>().at(0);
//
//    if (pottswt > 0) {
//      vp::ScopedTimer tmr("Adding 2D XYRGB pairwise terms .. ");
//
//      const StdVectorf& sigmas = vm.at("XYRGB.sigmas").as<StdVectorf>();
//
//      vgm.addSpatialPairwiseBilateral(sigmas.at(0), sigmas.at(1), sigmas.at(2), sigmas.at(3),
//                                      sigmas.at(4), images, vp::PottsFunction(pottswt),
//                                      vp::NORMALIZE_SYMMETRIC);
//    }
//  }
//
//  Eigen::Matrix<float, 2, Eigen::Dynamic> featuresXY;
//  {
//    vp::ScopedTimer tmr("Loading Features " + feat_filepath + " ... ");
//
//    Eigen::MatrixXf allFeaturesXY;
//    vp::FeatureInfo finfo;
//
//    std::tie(allFeaturesXY, finfo) = vp::readFeatures(feat_filepath);
//
//    if(finfo.dataset_name != dataset.name())
//      throw std::runtime_error("Feature is of wrong dataset");
//
//    if(start_frame < finfo.start_frame || ((start_frame + number_of_frames) > (finfo.start_frame + finfo.number_of_frames)))
//      throw std::runtime_error("Loaded Features doesn not have required frames");
//
//    if(allFeaturesXY.cols() < WH * number_of_frames) {
//      throw std::runtime_error("Loaded Features has wrong # of features");
//    }
//
//    featuresXY = allFeaturesXY.block(0, (start_frame - finfo.start_frame) * WH, 2,  number_of_frames * WH);
//  }
//
//  Eigen::RowVectorXf featuresT( W * H *number_of_frames);
//  {
//    for (int t = 0; t < number_of_frames; ++t) {
//      for (int p = 0; p < WH; ++p)
//        featuresT[t * WH + p] = t;
//    }
//  }
//
//  {
//    const float pottswt = vm.at("XYT.PottsWt").as<StdVectorf>().at(0);
//    if(pottswt > 0) {
//      vp::ScopedTimer tmr("Adding 3D XYT pairwise terms .. ");
//
//      const StdVectorf& sigmas = vm.at("XYT.sigmas").as<StdVectorf>();
//
//      Eigen::Matrix<float, 3, Eigen::Dynamic> features(3, W * H * number_of_frames);
//
//      features.row(0) = featuresXY.row(0) / sigmas.at(0);
//      features.row(1) = featuresXY.row(1) / sigmas.at(1);
//      features.row(2) = featuresT / sigmas.at(2);
//
//      typedef vp::StaticDenseKernel<3, vp::ConcurrentPermutohedralLattice> KernelType;
//      typedef vp::DensePairwiseFactor<KernelType, vp::PottsFunction> DensePairwisePottsFactor;
//
//      vgm.pairwise3D().push_back(
//          new DensePairwisePottsFactor(features, vp::PottsFunction(pottswt), vp::NORMALIZE_SYMMETRIC));
//    }
//  }
//
//  {
//
//    const float pottswt = vm.at("XYTRGB.PottsWt").as<StdVectorf>().at(0);
//
//    if(pottswt > 0) {
//      vp::ScopedTimer tmr("Adding 3D XYTRGB pairwise terms .. ");
//
//      const StdVectorf& sigmas = vm.at("XYTRGB.sigmas").as<StdVectorf>();
//
//      Eigen::Matrix<float, 6, Eigen::Dynamic> features(6, W * H *number_of_frames);
//
//      features.row(0) = featuresXY.row(0) / sigmas.at(0);
//      features.row(1) = featuresXY.row(1) / sigmas.at(1);
//      features.row(2) = featuresT / sigmas.at(2);
//      features.bottomRows<3>() = vp::createFeaturesRGB(sigmas.at(3), sigmas.at(4), sigmas.at(5), images);
//
//      typedef vp::StaticDenseKernel<6, vp::ConcurrentPermutohedralLattice> KernelType;
//      typedef vp::DensePairwiseFactor<KernelType, vp::PottsFunction> DensePairwisePottsFactor;
//
//      vgm.pairwise3D().push_back(new DensePairwisePottsFactor(features, vp::PottsFunction(pottswt), vp::NORMALIZE_SYMMETRIC));
//    }
//  }
//
//
//  return vgm;
//}
//
//int main(int argc, char **argv) {
//
//  namespace po = boost::program_options;
//
//  po::options_description generic_options("Generic Options");
//  generic_options.add_options()
//      ("help,h", "Help screen")
//      ;
//
//  po::options_description config_options("App Specific Config");
//  config_options.add_options()
//      ("feature_file,f",  po::value<std::string>()->required(), "Feature File")
//      ("iterations,i",  po::value<int>()->default_value(5), "Number of iterations")
//      ("display_save",  po::value<int>()->default_value(1), "0-DisplayOnly 1-Both 2-FileSaveOnly")
//      ("out_wildcard",  po::value<std::string>()->default_value("VideoGM_%06d.png"), "WildCard for Output Images")
//      ;
//
//  po::options_description cmdline_options;
//  cmdline_options.add(generic_options).add(config_options).add(vp::makeDatasetConfigOptions()).add(vp::makeKernelParamsOptions());
//
//
//  po::variables_map vm;
//
//  try {
//    po::store(parse_command_line(argc, argv, cmdline_options), vm);
//    po::notify(vm);
//    vp::verfyKernelParamsOptions(vm);
//  } catch (const po::error &ex) {
//    std::cerr << ex.what() << '\n';
//    std::cout << cmdline_options << '\n';
//    return EXIT_FAILURE;
//  }
//
//  if (vm.count("help")) {
//    std::cout << cmdline_options << '\n';
//    return EXIT_SUCCESS;
//  }
//
//
//  std::cout << "---------- Config -----------\n";
//  vp::printPOVariablesMap(vm);
//  std::cout << "-----------------------------\n";
//
//
//  // Setup STGM
//  vp::VideoGM stgm = setupGM(vm);
//
//  std::cout << "3D pairwise terms:\n";
//  for (const vp::PairwiseFactor& pairwise_factor : stgm.pairwise3D()) {
//    std::cout << "    " << pairwise_factor.printInfo() << "\n";
//  }
//
//
//  std::vector<Eigen::VectorXi> map_results;
//  {
//    const int iterations = vm.at("iterations").as<int>();
//    vp::ScopedTimer tmr("Running Inference (" + std::to_string(iterations) + " iterations)\n");
//
//    // Run Inference
//    map_results = stgm.computeMAPestimate(iterations);
//  }
//
//
//  std::vector<cv::Mat> map_images;
//  {
//    vp::ScopedTimer tmr("Generating Result Images .... ");
//
//    // Create Image results
//    map_images.resize(map_results.size());
//#pragma omp parallel for
//    for (int i = 0; i < map_images.size(); ++i) {
//      map_images[i] = vp::getRGBImageFromLabels(map_results[i], stgm.width(),
//                                                stgm.height(),
//                                                stgm.numberOfLabels());
//    }
//  }
//
//  const int display_save = vm["display_save"].as<int>();
//  const int start_frame = vm.at("start_frame").as<int>();
//
//  if(display_save > 0) {
//    vp::ScopedTimer tmr("Saving Map label images ..");
//    vp::saveImages(map_images, start_frame, vm.at("out_wildcard").as<std::string>());
//  }
//
//  if (display_save < 2) {
//    vp::visualizeImages(map_images, start_frame, "MapResults");
//  }
//
//
//  return EXIT_SUCCESS;
//}
//
//
//
