/**
 * @file visualizeMaxUnaries.cpp
 * @brief visualize Unary scores by taking pixelwise max
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/UnaryIO.h"
#include "VideoParsing/Utils/SemanticLabel.h"

#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

cv::Mat generateMaxUnaryImage(const std::string& filepath, const int width, const int height) {
  using std::cout;

  cv::Mat max_unary(height, width, CV_8UC3);

  Eigen::MatrixXf unary_data = vp::loadUnary(filepath);

  if(unary_data.cols() != (height * width)) {
    std::cout << "ERROR: Bad Size" << std::endl;
    return max_unary;
  }


  Eigen::DenseIndex pixel_idx = 0;
  for (int r = 0; r < max_unary.rows; ++r)
    for (int c = 0; c < max_unary.cols; ++c) {
      int max_loc = 0;
      unary_data.col(pixel_idx).maxCoeff(&max_loc);
      max_unary.at<cv::Vec3b>(r, c) = vp::labelToBGR(static_cast<vp::Camvid11>(max_loc));
      ++pixel_idx;
    }

  return max_unary;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "ERROR: image file path not provided\n";
    std::cout << "Usage: " << argv[0] << " width height /path/to/unary/files\n";
    return EXIT_FAILURE;
  }

  const int width = std::stoi(argv[1]);
  const int height = std::stoi(argv[2]);
  const int number_of_unaries = argc - 3;

  std::cout << "Image Dimension: " << width << " x " << height << "\n";
  std::cout << "User provided " << number_of_unaries << " input files\n";

  cv::namedWindow("max_unary");

  bool paused = true;
  for (int i = 0; i < number_of_unaries; ++i) {

    namespace bfs = boost::filesystem;

    bfs::path unary_fp(argv[i + 3]);

    if(!bfs::exists(unary_fp)) {
      std::cout << "ERROR: " << unary_fp << " does not exist\n";
      return EXIT_FAILURE;
    }

    if(!bfs::is_regular(unary_fp)) {
      std::cout << "ERROR: " << unary_fp << " is not a regular file\n";
      return EXIT_FAILURE;
    }

    cv::Mat max_unary = generateMaxUnaryImage(unary_fp.string(), width, height);
    cv::imshow("max_unary", max_unary);
    cv::displayOverlay("max_unary", "Image #" + std::to_string(i));


    std::string out_image_name = "max_unary_" + unary_fp.stem().string() + ".png";
    cv::imwrite(out_image_name, max_unary);

    char key = (char) cv::waitKey(!paused);
    if (key == 'q')
      break;
    else if(key == 'p') {
      paused = !paused;
    }
  }

  return EXIT_SUCCESS;
}
