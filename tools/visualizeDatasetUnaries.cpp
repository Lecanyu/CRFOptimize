/**
 * @file visualizeDatasetUnaries.cpp
 * @brief visualizeDatasetUnaries
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/Dataset.h"
#include "VideoParsing/IO/SequenceDataIO.h"
#include "VideoParsing/Utils/SemanticLabel.h"
#include "VideoParsing/Core/EigenUtils.h"
#include "VideoParsing/Core/EigenOpenCVUtils.h"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {

  if (argc < 4) {
    std::cout << "ERROR: parsing inputs\n";
    std::cout << "Usage: " << argv[0] << " config_file start_frame number_of_frames [UnaryWildCard]\n";
    return EXIT_FAILURE;
  }

  vp::Dataset dataset(argv[1]);
  dataset.print();

  const int start_frame = std::stoi(argv[2]);
  const int number_of_frames = std::stoi(argv[3]);

  const int W = dataset.imageWidth();
  const int H = dataset.imageHeight();

  std::vector<Eigen::MatrixXf> unaries;
  {
    std::cout << "Loading " << number_of_frames << " Unaries [" << start_frame << ", "
                << start_frame + number_of_frames - 1 << "] .... " << std::flush;

    if(argc > 4)
      unaries = vp::loadUnariesAsProb(argv[4], start_frame, number_of_frames);
    else
      unaries = dataset.loadUnariesAsProb(start_frame, number_of_frames);
    std::cout << "Done" << std::endl;
  }

  Eigen::Index M = 0;
  {
    // Get max label count
    for(const Eigen::MatrixXf& unary : unaries)
      M = std::max(M, unary.rows());
  }

  std::vector<std::string> window_names;

  if( M == 11) {
    window_names = {{
        "BUILDING",
        "TREE",
        "SKY",
        "CAR",
        "SIGN_SYMBOL",
        "ROAD",
        "PEDESTRIAN",
        "FENCE",
        "COLUMN_POLE",
        "SIDEWALK",
        "BICYCLIST"
    }};

    { // Set window names and size and position
      for (int i = 0; i < 11; ++i) {
        cv::namedWindow(window_names[i], cv::WINDOW_NORMAL);
        cv::resizeWindow(window_names[i], 480, 360);
        cv::moveWindow(window_names[i], (i % 4) * 480, (i / 4) * 360);
      }
      cv::namedWindow("max_unary", cv::WINDOW_NORMAL);
      cv::resizeWindow("max_unary", 480, 360);
      cv::moveWindow("max_unary", 3 * 480, 2 * 360);
    }
  }
  else {
    window_names.resize(M);
    for (Eigen::Index i =0 ;i< M; ++i) {
      window_names[i] = "Label " + std::to_string(i);
      cv::namedWindow(window_names[i], cv::WINDOW_NORMAL);
    }
    cv::namedWindow("max_unary", cv::WINDOW_NORMAL);
  }



  bool paused = true;
  int step = 1;
  for(int i = 0;;) {
    int frame_id = start_frame + i;
    cv::displayOverlay("max_unary", dataset.name() + "  " + std::to_string(frame_id));

    const Eigen::MatrixXf& unary = unaries[i];

    const int M = unary.rows();

    assert(unary.cols() == W*H);
    assert(unary.rows() == M);



    cv::imshow("max_unary", vp::getRGBImageFromLabels(vp::getColwiseMaxCoeffIndex(unary), W, H, M ));

    std::vector<cv::Mat> prob_images(M);
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      prob_images[i] = convertToImage32F(unary.row(i).transpose(), W, H);
    }

    for (int i = 0; i < M; ++i) {
      cv::imshow(window_names[i], prob_images[i]);
    }

    const int key = cv::waitKey(!paused);

    if (key == 27 || key == 81 || key == 113)  // Esc or Q or q
      break;
    else if (key == 123 || key == 125)  // Down or Left Arrow
      step = -1;
    else if (key == 124 || key == 126)  // Up or Right Arrow
      step = 1;
    else if (key == 'p' || key == 'P')
      paused = !paused;

    i += step;
    i = std::max(0, i);
    i = std::min(i, number_of_frames - 1);
  }

  return EXIT_SUCCESS;
}

