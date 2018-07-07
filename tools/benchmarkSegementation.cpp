/**
 * @file benchmarkSegementation.cpp
 * @brief code to evaluate segmentation against ground-truth
 *
 * Currently only supports the predefined static Camvid11 and CityScape19 SemantiCLabels
 * TODO: Support for arbitrary SemanticLabels
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Utils/SemanticLabel.h"
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {

  if (argc < 5) {
    std::cout << "ERROR: parsing inputs\n";
    std::cout << "Usage: " << argv[0] << " <NumOfLabels> <GTFolder> <ResultFolder> <SplitImageListFile>\n\n";
    std::cout << "Example For evaluating Camvid11 the command should be something like:\n";
    std::cout << argv[0] << " 11 folder/of/GTimages /folder/of/ResultImages ../config-files/Camvid_test_image_list.txt\n";
    return EXIT_FAILURE;
  }

  // number of labels
  const int M = std::stoi(argv[1]);
  std::cout << "Number of labels = " << M << "\n";

  namespace fs = boost::filesystem;

  fs::path gt_folder(argv[2]);
  if (!fs::exists(gt_folder) || !fs::is_directory(gt_folder)) {
    std::cout << "Provided GTFolder is invalid: " << gt_folder << "\n";
    return EXIT_FAILURE;
  }

  fs::path result_folder(argv[3]);
  if (!fs::exists(result_folder) || !fs::is_directory(result_folder)) {
    std::cout << "Provided ResultFolder is invalid: " << result_folder << "\n";
    return EXIT_FAILURE;
  }

  std::vector<std::string> image_names;
  {
    std::ifstream file(argv[4]);
    if (!file) {
      std::cout << "Error opening output file: " << argv[3] << std::endl;
      return EXIT_FAILURE;
    }

    for (std::string line; std::getline( file, line ); /**/ )
      image_names.push_back( line );

    file.close();
  }

  std::cout << "Evaluating on " << image_names.size() << " Images" << std::endl;


  int total_pixels = 0;
  int ok_pixels = 0;

  Eigen::VectorXi total_pixels_class = Eigen::VectorXi::Zero(M);
  Eigen::VectorXi ok_pixels_class = Eigen::VectorXi::Zero(M);
  Eigen::VectorXi label_pixels = Eigen::VectorXi::Zero(M);

  for (const std::string& image_name : image_names) {
    fs::path gt_image_fp = gt_folder / fs::path(image_name);
    if (!fs::exists(gt_image_fp) || !fs::is_regular_file(gt_image_fp)) {
      std::cout << "Invalid Image path: " << gt_image_fp << "\n";
      return EXIT_FAILURE;
    }

    fs::path result_image_fp = result_folder / fs::path(image_name);
    if (!fs::exists(result_image_fp) || !fs::is_regular_file(result_image_fp)) {
      std::cout << "Invalid Image path: " << result_image_fp << "\n";
      return EXIT_FAILURE;
    }

    cv::Mat gt_image = cv::imread(gt_image_fp.string());
    cv::Mat result_image = cv::imread(result_image_fp.string());

    if(gt_image.size != result_image.size) {
      std::cout << "ERROR: gt_image.size != result_image.size\n";
      return EXIT_FAILURE;
    }

    if(M==19) {
      for (int y = 0; y < gt_image.rows; ++y)
         for (int x = 0; x < gt_image.cols; ++x) {
           vp::CityScape19 gt_label = vp::bgrToCityScape19(gt_image.at<cv::Vec3b>(y, x));
           if(gt_label != vp::CityScape19::IGNORE) {
             ++total_pixels;
             ++total_pixels_class[static_cast<int>(gt_label)];

             vp::CityScape19 result_label = vp::bgrToCityScape19(result_image.at<cv::Vec3b>(y, x));
             ++label_pixels[static_cast<int>(result_label)];

             if(gt_label == result_label) {
               ++ok_pixels;
               ++ok_pixels_class[static_cast<int>(gt_label)];
             }
           }
         }
    }
    else if(M == 11) {
      for (int y = 0; y < gt_image.rows; ++y)
         for (int x = 0; x < gt_image.cols; ++x) {
           vp::Camvid11 gt_label = vp::bgrToCamvid11(gt_image.at<cv::Vec3b>(y, x));
           if(gt_label != vp::Camvid11::VOID) {
             ++total_pixels;
             ++total_pixels_class[static_cast<int>(gt_label)];

             vp::Camvid11 result_label = vp::bgrToCamvid11(result_image.at<cv::Vec3b>(y, x));
             ++label_pixels[static_cast<int>(result_label)];

             if(gt_label == result_label) {
               ++ok_pixels;
               ++ok_pixels_class[static_cast<int>(gt_label)];
             }
           }
         }
    }
    else {
      throw std::runtime_error("Currently only support Camvid11 or Cityscape19 labelspace");
    }


  }

  Eigen::VectorXd iou_scores = 100.0 * ok_pixels_class.cast<double>().array()
      / (total_pixels_class + label_pixels - ok_pixels_class).cast<double>().array();

  std::cout << "ClassIOU= " << iou_scores.transpose() << std::endl;
  std::cout << "MeanIOU= " << iou_scores.mean() << std::endl;
  std::cout << "PixelAccuracy= " << (ok_pixels * 100.0) / double (total_pixels) << std::endl;
}


