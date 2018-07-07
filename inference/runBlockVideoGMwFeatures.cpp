/**
 * @file runBlockVideoGMwFeatures.cpp
 * @brief runBlockVideoGMwFeatures
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/DenseGM/BlockVideoGM.h"
#include "VideoParsing/IO/Dataset.h"
#include "VideoParsing/IO/FeatureIO.h"
#include "VideoParsing/Core/CoordToIndex.h"
#include "VideoParsing/Utils/SemanticLabel.h"
#include "VideoParsing/Utils/FilterHelper.h"
#include "VideoParsing/Utils/ScopedTimer.h"
#include "VideoParsing/Utils/VisualizationHelper.h"
#include "VideoParsing/Utils/ProgramOptionsHelper.h"

#include <boost/assert.hpp>
#include <iostream>


vp::BlockVideoGM setupGM(const boost::program_options::variables_map& vm) {

  vp::Dataset dataset(vm.at("dataset").as<std::string>());
  dataset.print();

  const int start_frame = vm.at("start_frame").as<int>();
  const int number_of_frames = vm.at("number_of_frames").as<int>();
  const std::vector<std::string> feat_filepaths = vm.at("feature_files").as<std::vector<std::string>>();

  const int W = dataset.imageWidth();  // width of the grid
  const int H = dataset.imageHeight();  // height of the grid
  const int M = dataset.loadUnaryAsNegLogProb(start_frame).rows(); // Number of labels
  const int WH = W * H;


  vp::BlockVideoGM vgm(W, H, number_of_frames, M);

  {
    vp::ScopedTimer tmr("Loading and Adding Unaries ..");
    std::vector<Eigen::MatrixXf> unaries = dataset.loadUnariesAsProb(start_frame, number_of_frames);

#pragma omp parallel for
    for (int i = 0; i < number_of_frames; ++i) {
      Eigen::MatrixXf& unary = unaries[i];

      // This makes sure unary prob are not zero
      unary.array() += 0.0001f;
      unary = unary.array().rowwise() / unary.colwise().sum().eval().array();

      // Take negative log prob (This is mandatory)
      unary = -unary.array().log();
    }

    vgm.addUnaries(unaries);
    assert(vgm.spatialGMs().size() == number_of_frames);
  }

  std::vector<cv::Mat> images;
  {
    vp::ScopedTimer tmr("Loading Images ..");
    images = dataset.loadImageSequence(start_frame, number_of_frames);
    for (int i = 0; i < number_of_frames; ++i) {
      const cv::Mat& image = images[i];
      BOOST_VERIFY(image.cols == W);
      BOOST_VERIFY(image.rows == H);
    }
  }


  typedef std::vector<float> StdVectorf;

  // Add spatial only factors
  {
    const float pottswt = vm.at("XY.PottsWt").as<StdVectorf>().at(0);
    if(pottswt > 0) {
      vp::ScopedTimer tmr("Adding 2D XY pairwise terms .. ");
      const StdVectorf& sigmas = vm.at("XY.sigmas").as<StdVectorf>();
      vgm.addSpatialPairwiseGaussian(sigmas.at(0), sigmas.at(1),
                                     vp::PottsFunction(pottswt),
                                     vp::NORMALIZE_SYMMETRIC);
    }
  }


  {
    const float pottswt = vm.at("XYRGB.PottsWt").as<StdVectorf>().at(0);

    if (pottswt > 0) {
      vp::ScopedTimer tmr("Adding 2D XYRGB pairwise terms .. ");

      const StdVectorf& sigmas = vm.at("XYRGB.sigmas").as<StdVectorf>();

      vgm.addSpatialPairwiseBilateral(sigmas.at(0), sigmas.at(1), sigmas.at(2), sigmas.at(3),
                                      sigmas.at(4), images, vp::PottsFunction(pottswt),
                                      vp::NORMALIZE_SYMMETRIC);
    }
  }


  {
    const float pottswt = vm.at("XYRGBUV.PottsWt").as<StdVectorf>().at(0);

    if (pottswt > 0) {
      vp::ScopedTimer tmr("Adding 2D XYRGBUV pairwise terms .. ");

      const StdVectorf& sigmas = vm.at("XYRGBUV.sigmas").as<StdVectorf>();

      // Make sure this is single threaded kernel
      typedef vp::StaticDenseKernel<7> KernelType;
      typedef vp::DensePairwiseFactor<KernelType, vp::PottsFunction> DensePairwisePottsFactor;

#pragma omp parallel for
      for (int i = 0; i < vgm.spatialGMs().size(); ++i) {
        const int frame_id = start_frame + i;

        Eigen::Matrix<float, 7, Eigen::Dynamic> features = vp::createFeaturesXYRGBUV(
            sigmas[0], sigmas[1], sigmas[2], sigmas[3], sigmas[4], sigmas[5], sigmas[6], images[i],
            dataset.loadFwdOpticalFlow(frame_id));

        vgm.spatialGMs()[i].pairwise().push_back(
            new DensePairwisePottsFactor(features, vp::PottsFunction(pottswt),vp::NORMALIZE_SYMMETRIC));
      }
    }
  }

  Eigen::Matrix<float, 3, Eigen::Dynamic> featuresRGB = vp::createFeaturesRGB(1, 1, 1, images);
  Eigen::RowVectorXf featuresT(W * H * number_of_frames);
  {
    for (int t = 0; t < number_of_frames; ++t) {
      for (int p = 0; p < WH; ++p)
        featuresT[t * WH + p] = t;
    }
  }


  for (const std::string& feat_filepath : feat_filepaths) {

    std::cout << "--------------------------------------------------\n";

    Eigen::Matrix<float, 2, Eigen::Dynamic> featuresXY;
    vp::FeatureInfo finfo;
    {
      vp::ScopedTimer tmr("Loading Features " + feat_filepath + " ... ");
      std::tie(featuresXY, finfo) = vp::readFeatures(feat_filepath);

      if (finfo.dataset_name != dataset.name())
        throw std::runtime_error("Feature is of wrong dataset");

      if (finfo.start_frame < start_frame)
        throw std::runtime_error("finfo.start_frame < start_frame");

      if ((finfo.start_frame + finfo.number_of_frames)
          > (start_frame + number_of_frames))
        throw std::runtime_error("finfo.last_frame >  last_frame");
    }

    std::cout << "Adding pairwise term over [" << finfo.start_frame << ", "
              << finfo.start_frame + finfo.number_of_frames - 1 << "]\n";

    const int relative_start_frame_id = (finfo.start_frame - start_frame);
    const Eigen::Index start_index = WH * relative_start_frame_id;
    const Eigen::Index nunber_of_indices = WH * finfo.number_of_frames;

    {
      const float pottswt = vm.at("XYT.PottsWt").as<StdVectorf>().at(0);
      if (pottswt > 0) {
        vp::ScopedTimer tmr("Adding 3D XYT pairwise terms .. ");

        const StdVectorf& sigmas = vm.at("XYT.sigmas").as<StdVectorf>();

        Eigen::Matrix<float, 3, Eigen::Dynamic> features(3, W * H * finfo.number_of_frames);

        features.row(0) = featuresXY.row(0) / sigmas.at(0);
        features.row(1) = featuresXY.row(1) / sigmas.at(1);
        features.row(2) = featuresT.segment(start_index, nunber_of_indices) / sigmas.at(2);

        typedef vp::StaticDenseKernel<3, vp::ConcurrentPermutohedralLattice> KernelType;
        typedef vp::DensePairwiseFactor<KernelType, vp::PottsFunction> DensePairwisePottsFactor;

        vgm.addPairwise3D(
            new DensePairwisePottsFactor(features, vp::PottsFunction(pottswt),
                                         vp::NORMALIZE_SYMMETRIC),
                                         relative_start_frame_id, finfo.number_of_frames);
      }
    }

    {

      const float pottswt = vm.at("XYTRGB.PottsWt").as<StdVectorf>().at(0);

      if (pottswt > 0) {
        vp::ScopedTimer tmr("Adding 3D XYTRGB pairwise terms .. ");

        const StdVectorf& sigmas = vm.at("XYTRGB.sigmas").as<StdVectorf>();

        Eigen::Matrix<float, 6, Eigen::Dynamic> features(6, W * H * finfo.number_of_frames);

        features.row(0) = featuresXY.row(0) / sigmas.at(0);
        features.row(1) = featuresXY.row(1) / sigmas.at(1);
        features.row(2) = featuresT.segment(start_index, nunber_of_indices) / sigmas.at(2);

        features.row(3) = featuresRGB.row(0).segment(start_index, nunber_of_indices) / sigmas.at(3);
        features.row(4) = featuresRGB.row(1).segment(start_index, nunber_of_indices) / sigmas.at(4);
        features.row(5) = featuresRGB.row(2).segment(start_index, nunber_of_indices) / sigmas.at(5);

        typedef vp::StaticDenseKernel<6, vp::ConcurrentPermutohedralLattice> KernelType;
        typedef vp::DensePairwiseFactor<KernelType, vp::PottsFunction> DensePairwisePottsFactor;

        vgm.addPairwise3D(
            new DensePairwisePottsFactor(features, vp::PottsFunction(pottswt),
                                         vp::NORMALIZE_SYMMETRIC),
                                         relative_start_frame_id, finfo.number_of_frames);
      }
    }
  }


  if (vm.count("HOTrack.all_not_same")) {
    vp::ScopedTimer tmr("Adding HO Track Factors .. ");
    // Apply track Factors
    std::vector<vp::Track2D> tracks = dataset.loadTracks(start_frame, number_of_frames);

    const vp::CoordToIndex3i coordToindex(W, H, number_of_frames);


    for (const vp::Track2D& track : tracks) {
      const int track_start_frame_id = std::max(start_frame, track.startFrameId());
      if(track_start_frame_id >= (start_frame + number_of_frames))
        continue;

      const int track_end_frame_id = std::min(start_frame + number_of_frames, track.endFrameId());
      if (track_end_frame_id <= start_frame)
        continue;


      std::vector<int> track_point_ids;
      for(int frame_id = track_start_frame_id; frame_id < track_end_frame_id; ++frame_id) {
        assert(track.isActive(frame_id));

        const Eigen::Vector2i loc = track.locationAtFrame(frame_id).array().round().cast<int>();

        const int index = coordToindex(loc.x(), loc.y(), frame_id - start_frame);
        track_point_ids.push_back(index);
      }
      vgm.hoTrackFactors().emplace_back(track_point_ids,
                                        vm["HOTrack.all_same"].as<float>(),
                                        vm["HOTrack.all_not_same"].as<float>());
    }
  }


  std::cout << "--------------------------------------------------\n";
  std::cout << "--------------------------------------------------\n";
  std::cout << "GM has now been constructed\n";
  return vgm;
}

int main(int argc, char **argv) {

  namespace po = boost::program_options;

  po::options_description generic_options("Generic Options");
  generic_options.add_options()
      ("help,h", "Help screen")
      ;

  po::options_description config_options("App Specific Config");
  config_options.add_options()
      ("feature_files,f",  po::value<std::vector<std::string>>()->multitoken()->required(), "Feature Files")
      ("iterations,i",  po::value<int>()->default_value(5), "Number of iterations")
      ("display_save",  po::value<int>()->default_value(1), "0-DisplayOnly 1-Both 2-FileSaveOnly")
      ("out_wildcard",  po::value<std::string>()->default_value("BlockVideoGM_%06d.png"), "WildCard for Output Images")
      ;

  po::options_description cmdline_options;
  cmdline_options.add(generic_options).add(config_options).add(vp::makeDatasetConfigOptions()).add(
      vp::makeKernelParamsOptions()).add(vp::makeHOSegementPotentialOptions()).add(
      vp::makeHOTrackPotentialOptions());


  po::variables_map vm;

  try {
    po::store(parse_command_line(argc, argv, cmdline_options), vm);
    po::notify(vm);
    vp::verfyKernelParamsOptions(vm);
  } catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  if (vm.count("help")) {
    std::cout << cmdline_options << '\n';
    return EXIT_SUCCESS;
  }


  std::cout << "---------- Config -----------\n";
  vp::printPOVariablesMap(vm);
  std::cout << "-----------------------------\n";


  // Setup STGM
  vp::BlockVideoGM stgm = setupGM(vm);

  std::cout << "3D pairwise terms:\n";
  for (const vp::PairwiseFactor& pairwise_factor : stgm.pairwise3D()) {
    std::cout << "    " << pairwise_factor.printInfo() << "\n";
  }

  std::cout << "# of HO Track Factors = " << stgm.hoTrackFactors().size() << "\n";


  std::vector<Eigen::VectorXi> map_results;
  {
    const int iterations = vm.at("iterations").as<int>();
    vp::ScopedTimer tmr("Running Inference (" + std::to_string(iterations) + " iterations)\n");

    // Run Inference
    map_results = stgm.computeMAPestimate(iterations);
  }


  std::vector<cv::Mat> map_images;
  {
    vp::ScopedTimer tmr("Generating Result Images .... ");

    // Create Image results
    map_images.resize(map_results.size());
#pragma omp parallel for
    for (int i = 0; i < map_images.size(); ++i) {
      map_images[i] = vp::getRGBImageFromLabels(map_results[i], stgm.width(),
                                                stgm.height(),
                                                stgm.numberOfLabels());
    }
  }

  const int display_save = vm["display_save"].as<int>();
  const int start_frame = vm.at("start_frame").as<int>();

  if(display_save > 0) {
    vp::ScopedTimer tmr("Saving Map label images ..");
    vp::saveImages(map_images, start_frame, vm.at("out_wildcard").as<std::string>());
  }

  if (display_save < 2) {
    vp::visualizeImages(map_images, start_frame, "MapResults");
  }

  return EXIT_SUCCESS;
}



