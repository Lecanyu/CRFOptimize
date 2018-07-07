/**
 * @file runFSO_XY.cpp
 * @brief runs features space optimization (Requires images, tracks, flow, edges(optional))
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/Dataset.h"
#include "VideoParsing/Core/EigenOpenCVUtils.h"
#include "VideoParsing/Utils/ProgramOptionsHelper.h"
#include "VideoParsing/IO/FeatureIO.h"
#include "VideoParsing/IO/TracksIO.h"
#include "VideoParsing/Features/VisualizeFeatures.h"
#include "VideoParsing/Features/LinearSolver.h"
#include "VideoParsing/Features/FeatureHelper.h"
#include "VideoParsing/Features/LinearSolverHelper.h"
#include "VideoParsing/Features/DifferentialOperators.h"

#include <Eigen/Sparse>
#include <Eigen/SparseExtra>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <iostream>


template <class VectorType>
VectorType makeEdgeProbVector(const vp::Dataset& dataset, const int start_frame, const int number_of_frames) {
  typedef typename VectorType::Scalar Scalar;
  const int WH = dataset.imageWidth() * dataset.imageHeight();
  VectorType edge_probs(WH * number_of_frames);

  for (int i = 0; i < number_of_frames; ++i) {
    edge_probs.segment(i * WH, WH) = dataset.loadEdgeProb(i + start_frame).cast<Scalar>();
  }

  return edge_probs;
}

template<class Scalar>
void addLaplacianCoeffsN8(std::vector<Eigen::Triplet<Scalar>>& coeffs,
                          const std::vector<cv::Mat>& images) {
  assert(images.size() > 0);
  const int W = images.front().cols;
  const int H = images.front().rows;
  const int T = images.size();

  for (int t = 0; t < T; ++t) {
    assert(images.at(t).channels() == 3);
    assert(images.at(t).depth() == CV_32F);
  }

  const vp::CoordToIndex3i coordToindex(W, H, T);

  for (int t = 0; t < T; ++t) {
    const cv::Mat& image = images[t];
    for (int my = 0; my < H; ++my)
      for (int mx = 0; mx < W; ++mx) {
        const int midx = coordToindex(mx, my, t);

        std::array<int, 9> col_idxs;
        Eigen::Matrix<Scalar, 9, 1> gvals;

        int nn = 0;
        for (int y = std::max<int>(0, my - 1); y <= std::min<int>(H - 1, my + 1); ++y)
          for (int x = std::max<int>(0, mx - 1); x <= std::min<int>(W - 1, mx + 1); ++x) {
            if (y == my && x == mx)
              continue;

            col_idxs[nn] = coordToindex(x, y, t);
            gvals[nn] = image.at<cv::Vec3f>(y, x)[0];  // Get Y values only
            ++nn;
          }

        assert(nn < 9);

        Scalar tval = image.at<cv::Vec3f>(my, mx)[0];  // Get Y values only
        gvals[nn] = tval;

        Scalar c_var = (gvals.head(nn + 1).array() - gvals.head(nn + 1).mean()).square().mean();
        Scalar c_sig = c_var * 0.6;
        Scalar mgv = (gvals.head(nn).array() - tval).square().minCoeff();

        c_sig = std::max<Scalar>(c_sig, -mgv / std::log(Scalar(0.01)));
        c_sig = std::max<Scalar>(c_sig, Scalar(0.000002));

        gvals.head(nn) = (-(gvals.head(nn).array() - tval).square() / c_sig).exp();
        gvals.head(nn) = gvals.head(nn).array() / gvals.head(nn).sum();

        for (int i = 0; i < nn; ++i)
          coeffs.emplace_back(midx, col_idxs[i], -gvals[i]);

        coeffs.emplace_back(midx, midx, 1);
      }
  }
}

template<class SparseMatrixType>
SparseMatrixType createLaplacianN8(const std::vector<cv::Mat>& rgb_images, double spatial_smoothness_wt = 1.0) {
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef Eigen::Triplet<Scalar> TripletType;

  const int W = rgb_images.front().cols;
  const int H = rgb_images.front().rows;
  const int T = rgb_images.size();
  const int WHT = W * H * T;

  SparseMatrixType L(WHT, WHT);

  {
    std::vector<cv::Mat> yuv_images(T);
    for (std::size_t i = 0; i < rgb_images.size(); ++i) {
      rgb_images[i].convertTo(yuv_images[i], CV_32FC3);
      yuv_images[i] *= 1. / 255;
      cv::cvtColor(yuv_images[i], yuv_images[i], CV_BGR2YCrCb);
    }

    std::vector<TripletType> Lcoeffs;
    addLaplacianCoeffsN8(Lcoeffs, yuv_images);
    L.setFromTriplets(Lcoeffs.begin(), Lcoeffs.end());
  }

  L *= spatial_smoothness_wt;

  return L;
}

template<class Scalar>
void addLaplacianCoeffsN4(std::vector<Eigen::Triplet<Scalar>>& coeffs,
                          const std::vector<cv::Mat>& rgb_images, double lambda = 1.0,
                          double color_sigma = 10) {
  const int W = rgb_images.front().cols;
  const int H = rgb_images.front().rows;
  const int T = rgb_images.size();

  assert(rgb_images.front().type() == CV_8UC3);

  const vp::CoordToIndex3i coordToindex(W, H, T);
  for (int t = 0; t < T; ++t) {
    const cv::Mat& image = rgb_images[t];

    for (int my = 0; my < H; ++my)
      for (int mx = 0; mx < W; ++mx) {
        const int midx = coordToindex(mx, my, t);

        typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
        Vector3 mrgb = convertVec3b<Vector3>(image.at < cv::Vec3b > (my, mx));

        Scalar total_wt = 0;
        if ((mx - 1) >= 0) {
          Vector3 diff = (convertVec3b<Vector3>(image.at < cv::Vec3b > (my, mx - 1)) - mrgb)
              / color_sigma;
          Scalar wt = std::exp(-0.5 * diff.squaredNorm());
          total_wt += wt;
          coeffs.emplace_back(midx, coordToindex(mx - 1, my, t), -lambda * wt);
        }
        if ((mx + 1) < W) {
          Vector3 diff = (convertVec3b<Vector3>(image.at < cv::Vec3b > (my, mx + 1)) - mrgb)
              / color_sigma;
          Scalar wt = std::exp(-0.5 * diff.squaredNorm());
          total_wt += wt;
          coeffs.emplace_back(midx, coordToindex(mx + 1, my, t), -lambda * wt);
        }
        if ((my - 1) >= 0) {
          Vector3 diff = (convertVec3b<Vector3>(image.at < cv::Vec3b > (my - 1, mx)) - mrgb)
              / color_sigma;
          Scalar wt = std::exp(-0.5 * diff.squaredNorm());
          total_wt += wt;
          coeffs.emplace_back(midx, coordToindex(mx, my - 1, t), -lambda * wt);
        }
        if ((my + 1) < H) {
          Vector3 diff = (convertVec3b<Vector3>(image.at < cv::Vec3b > (my + 1, mx)) - mrgb)
              / color_sigma;
          Scalar wt = std::exp(-0.5 * diff.squaredNorm());
          total_wt += wt;
          coeffs.emplace_back(midx, coordToindex(mx, my + 1, t), -lambda * wt);
        }

        coeffs.emplace_back(midx, midx, lambda * total_wt);
      }
  }
}


template<class SparseMatrixType>
SparseMatrixType createLaplacianN4(const std::vector<cv::Mat>& rgb_images, double spatial_smoothness_wt = 1.0, double color_sigma = 10.0) {
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef Eigen::Triplet<Scalar> TripletType;

  const int W = rgb_images.front().cols;
  const int H = rgb_images.front().rows;
  const int T = rgb_images.size();
  const int WHT = W * H * T;

  SparseMatrixType L(WHT, WHT);
  {
    std::vector<TripletType> Lcoeffs;
    addLaplacianCoeffsN4(Lcoeffs, rgb_images, spatial_smoothness_wt, color_sigma);
    L.setFromTriplets(Lcoeffs.begin(), Lcoeffs.end());
  }

  return L;
}

int main(int argc, char **argv) {

  namespace po = boost::program_options;

  po::options_description generic_options("Generic Options");
  generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("App Specific Config");
  config_options.add_options()
      ("fix_frame_id,f",  po::value<int>(), "FrameId of reference image. Defaults to middle")
      ("max_iterations,i",  po::value<int>()->default_value(100), "Max # of iterations")
      ("visualize,v",  po::value<int>()->default_value(1), "1:VizStyleA 2:VizStyleA Other:Nothing")
      ("tracks_file,t",  po::value<std::string>(), "Specify a custom tracks file")
      ;

  po::options_description cmdline_options;
  cmdline_options.add(generic_options).add(config_options).add(
      vp::makeDatasetConfigOptions());

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


  vp::Dataset dataset(vm.at("dataset").as<std::string>());
  dataset.print();

  const int start_frame = vm.at("start_frame").as<int>();
  const int T = vm.at("number_of_frames").as<int>();
  const int fix_frame_id =
      vm.count("fix_frame_id") ?
          vm.at("fix_frame_id").as<int>() : (start_frame + T / 2);
  const int last_frame_id = start_frame + T - 1;

  const int max_num_of_iterations = vm.at("max_iterations").as<int>();
  const int vizualize = vm.at("visualize").as<int>();




  std::cout << "Dataset = " << dataset.name() << "\n";
  std::cout << "Frames = [" << start_frame << "," << last_frame_id << "]\n";
  std::cout << "fix_frame_id = " << fix_frame_id << "\n";
  std::cout << "max_num_of_iterations = " << max_num_of_iterations << "\n";
  std::cout << "Visualize = " << vizualize << "\n";

  if ((fix_frame_id < start_frame) || (fix_frame_id > last_frame_id)) {
    std::cout << "Fixed FrameID is outside range\n";
    return EXIT_FAILURE;
  }


  typedef double Scalar;
  typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrixType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DenseVectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 2> DenseMatrixTypeX2;


  const int W = dataset.imageWidth();
  const int H = dataset.imageHeight();
  const int WH = W * H;
  const int WHT = W * H * T;

  std::cout << "Image Dimension = " << W << " X " << H << " (" << WH << " pixels).\n";
  std::cout << "Total number of pixels = " << WHT << "\n";

  const Scalar spatial_smoothness_wt = 0.1;
  const Scalar flow_terms_wt = 1.0;
//  const Scalar track_terms_wt = 0.05;

  const Scalar data_terms_wt = 500.0;
  const Scalar track_data_terms_wt = 50.0;



  SparseMatrixType A(WHT, WHT);

  // Add spatial smoothness terms
//  {
//    vp::ScopedTimer tmr("Adding Spatial Smoothness (D2) terms ... ");
//    A += spatial_smoothness_wt * vp::createD2TD2<SparseMatrixType>(W, H, T);
//  }
//  {
//    vp::ScopedTimer tmr("Adding Spatial Smoothness (L) terms ... ");
//    A += spatial_smoothness_wt * vp::createLaplacian<SparseMatrixType>(W, H, T);
//  }

//  //   Add spatial smoothness terms
//    {
//      vp::ScopedTimer tmr("Adding Spatial Smoothness terms ... ");
//      std::vector<cv::Mat> rgb_images = dataset.loadImageSequence(start_frame, T);
////      A += createLaplacianN8<SparseMatrixType>(rgb_images, spatial_smoothness_wt);
//      A += createLaplacianN4<SparseMatrixType>(rgb_images, spatial_smoothness_wt);
//    }

//  {
//    std::cout << "Computing Edge Weights ..  " << std::endl;
//    DenseVectorType edges = makeEdgeProbVector<DenseVectorType>(dataset, start_frame, T);
//
//    std::cout << "Edge Probs:= [" << edges.minCoeff() << " -- " << edges.maxCoeff() << "]\n";
//    double mean = edges.mean();
//    std::cout << "Edge Prob: Mean = " << mean << " Variance = " << edges.array().square().mean() - mean*mean << "\n";
//
//    edges = (-edges/0.03).array().exp();
//    std::cout << "Edge Weights:= [" << edges.minCoeff() << " -- " << edges.maxCoeff() << "]\n";
//
//
//    vp::ScopedTimer tmr("Adding Edge Sensitive Spatial Smoothness (L) terms ... ");
//    A += spatial_smoothness_wt * vp::createLaplacian<SparseMatrixType>(edges.asDiagonal(), W, H, T);
//  }

  {
    std::cout << "Computing Edge Weights ..  " << std::endl;
    DenseVectorType edges = makeEdgeProbVector<DenseVectorType>(dataset, start_frame, T);

    std::cout << "Edge Probs:= [" << edges.minCoeff() << " -- " << edges.maxCoeff() << "]\n";
    double mean = edges.mean();
    std::cout << "Edge Prob: Mean = " << mean << " Variance = " << edges.array().square().mean() - mean*mean << "\n";

    edges = (-edges/0.03).array().exp();
    std::cout << "Edge Weights:= [" << edges.minCoeff() << " -- " << edges.maxCoeff() << "]\n";


    vp::ScopedTimer tmr("Adding Edge Sensitive Spatial Smoothness (D2) terms ... ");
    A += spatial_smoothness_wt * vp::createD2TD2<SparseMatrixType>(edges.asDiagonal(), W, H, T);
  }

//   Add Temporal constraints via Fwd Flow
  {
    std::cout << "Loading Fwd Flows ... " << std::flush;
    std::vector<cv::Mat> fwd_flows = dataset.loadForwardFlows(start_frame, T - 1);
    std::cout << "Done" << std::endl;

    vp::ScopedTimer tmr("Adding Temporal constraints from Fwd Flows ... ");
    A += flow_terms_wt * vp::createGramMatrixFromFlows<SparseMatrixType>(fwd_flows, true);
  }
//
//  //   Add Temporal constraints via Bwd Flow
//  {
//    std::cout << "Loading Bwd Flows ... " << std::flush;
//    std::vector<cv::Mat> bwd_flows = dataset.loadBackwardFlows(start_frame, T - 1);
//    std::cout << "Done" << std::endl;
//
//    vp::ScopedTimer tmr("Adding Temporal constraints from Bwd Flows ... ");
//    A += flow_terms_wt * vp::createGramMatrixFromFlows<SparseMatrixType>(bwd_flows, false);
//  }

//  {
//    std::cout << "Loading Fwd Flows ... " << std::flush;
//    std::vector<cv::Mat> fwd_flows = dataset.loadForwardFlows(start_frame, T - 1);
//    std::cout << "Done" << std::endl;
//
//    std::cout << "Loading bwd Flows ... " << std::flush;
//    std::vector<cv::Mat> bwd_flows = dataset.loadBackwardFlows(start_frame, T - 1);
//    std::cout << "Done" << std::endl;
//
//    vp::ScopedTimer tmr("Adding Temporal constraints from Fwd/bwd Flows ... ");
//    A += flow_terms_wt * vp::createGramMatrixFromFlows<SparseMatrixType>(fwd_flows, bwd_flows, 10.0);
//  }


  std::vector<vp::Track2D> tracks;
  {
    vp::ScopedTimer tmr("Loading Tracks .. ");
    if(vm.count("tracks_file")) {
      namespace bf = boost::filesystem;
      bf::path track_fp(vm.at("tracks_file").as<std::string>());
      if (!bf::exists(track_fp)) {
        std::cout << "Tracks File: "<< track_fp << " does not exist\n";
        throw std::runtime_error("Tracks File does not exist at ");
      }
      tracks = vp::readTracks(track_fp.string());
    }
    else
      tracks = dataset.loadTracks(start_frame, T);
    std::cout << " Loaded " << tracks.size() << " tracks ";
  }

//  // Add Temporal constraints via Tracks
//  {
//    vp::ScopedTimer tmr("Adding Temporal constraints from Tracks ... ");
//    A += track_terms_wt * vp::createTracksLaplacian < SparseMatrixType > (tracks, start_frame, T, W, H);
//  }


  DenseMatrixTypeX2 B(WHT, 2);
  B.setZero();

  // Add Data terms (Equality Constraints coming from the fixed frame and tracks)
  {
    vp::ScopedTimer tmr("Adding Equality Constraints ... ");
    const int i = fix_frame_id - start_frame;
    assert(i>=0 && i < T);

    const vp::CoordToIndex3i coordToindex(W, H, T);

    // Add data terms for all pixels in fixed frame
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const int index = coordToindex(x, y, i);

        A.coeffRef(index, index) += data_terms_wt;

        B(index, 0) += data_terms_wt * x;
        B(index, 1) += data_terms_wt * y;
      }

    // Spread data terms via tracks
    for (const vp::Track2D& track : tracks) {
      if (track.isActive(fix_frame_id)) {
        const Eigen::Vector2i floc = track.locationAtFrame(fix_frame_id).array().round().cast<int>();

        for(int t = 0; t< T; ++t) {
          const int frame_id = start_frame + t;
          if(track.isActive(frame_id)) {
            const Eigen::Vector2i loc = track.locationAtFrame(frame_id).array().round().cast<int>();
            const int index = coordToindex(loc.x(), loc.y(), t);
            A.coeffRef(index, index) += track_data_terms_wt;
            B(index, 0) += track_data_terms_wt * floc.x();
            B(index, 1) += track_data_terms_wt * floc.y();
          }
        }

      }
    }
  }

  {
    // clear the tracks since it will no longer be used
    tracks.clear();
  }


  // Make A compressed (standard CRS format)
  {
    vp::ScopedTimer tmr("Compressing A matrix .. ");
    A.makeCompressed();
  }


  std::cout << "A.dimension() = " << A.rows() << " X " << A.cols() << "\n";
  std::cout << "A.nonZeros() = " << A.nonZeros() << " (" << (A.nonZeros() * 100.0) / A.size() << " %)\n";


//  DenseMatrixTypeX2 X = vp::solveViaCholesky(A, B);
//  DenseMatrixTypeX2 X = vp::solveViaCG(A, B, vp::computeFeaturesXY(W, H, T), max_num_of_iterations);
  DenseMatrixTypeX2 X = vp::solveViaAMGCL(A, B, vp::computeFeaturesXY(W, H, T));

  std::cout << "Min =" << X.colwise().minCoeff() << "\n";
  std::cout << "Max =" << X.colwise().maxCoeff() << "\n";

  Eigen::Matrix<float, 2, Eigen::Dynamic> featuresXY = X.transpose().cast<float>();

  {
    std::string out_fp = (boost::format("VWFXY-%s-%06d-%06d-%06d.feat") % dataset.name() % start_frame % last_frame_id % fix_frame_id).str();
    vp::ScopedTimer tmr("Saving Features as \"" + out_fp + "\" .... ");
    vp::saveFeatures(featuresXY, dataset, start_frame, out_fp);
  }

//  {
//    vp::ScopedTimer tmr("Exporting A matrix ... ");
//    Eigen::saveMarket(A, "A.mkt");
//  }

  if(vizualize == 1)
    vp::visualizeFeaturesXY(featuresXY, dataset, start_frame, T);
  else if(vizualize == 2)
    vp::visualizeFeaturesXYwShifting(featuresXY, dataset, start_frame, T);

  return EXIT_SUCCESS;

}
