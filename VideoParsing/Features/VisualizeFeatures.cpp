/**
 * @file VisualizeFeatures.cpp
 * @brief VisualizeFeatures
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Features/VisualizeFeatures.h"
#include "VideoParsing/Utils/VisualizationHelper.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace vp {

void visualizeFeaturesXY(const Eigen::Matrix<float, 2, Eigen::Dynamic>& featuresXY,
                         const Dataset& dataset, int start_frame, int number_of_frames,
                         const std::string& feature_name) {

  const int W = dataset.imageWidth();
  const int H = dataset.imageHeight();

  std::cout << "Loading " << number_of_frames << " Images [" << start_frame << ", "
            << start_frame + number_of_frames - 1 << "] .... " << std::flush;

  std::vector<cv::Mat> images = dataset.loadImageSequence(start_frame, number_of_frames);

  std::cout << "Done" << std::endl;

  const std::string windowname = "FeaturesXY: " + feature_name;
  cv::namedWindow(windowname);

  vp::WaitKeyNavigation navigator(0, number_of_frames - 1, true);
  do {
    const int i = navigator.value();

    cv::Mat flow_field_img = cv::Mat::zeros(H, W, CV_8UC3);

#pragma omp parallel for
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        Eigen::Index p = i * W * H + y * W + x;
        int fx = std::round(featuresXY(0, p));
        int fy = std::round(featuresXY(1, p));

        if (fx >= 0 && fx < W && fy >= 0 && fy < H)
          flow_field_img.at<cv::Vec3b>(fy, fx) = images.at(i).at<cv::Vec3b>(y, x);
      }

    cv::imshow(windowname, flow_field_img);
    cv::displayOverlay(windowname, "Frame " + std::to_string(i + start_frame));

  } while (navigator());
}

void visualizeFeaturesXYwShifting(Eigen::Matrix<float, 2, Eigen::Dynamic>& featuresXY,
                                  const Dataset& dataset, int start_frame, int number_of_frames,
                                  const std::string& feature_name) {

  const int W = dataset.imageWidth();
  const int H = dataset.imageHeight();

  int Wmax, Hmax;
  {
    Eigen::Vector2f feature_min = featuresXY.rowwise().minCoeff();
    Eigen::Vector2f feature_max = featuresXY.rowwise().maxCoeff();

    std::cout << "minimum: " << feature_min.transpose() << std::endl;
    std::cout << "maximum: " << feature_max.transpose() << std::endl;

    feature_min = feature_min.array().floor();

    featuresXY.colwise() -= feature_min;
    feature_max -= feature_min;

    feature_max = feature_max.array().ceil();

    std::cout << "Updated maximum: " << feature_max.transpose() << std::endl;
    Wmax = feature_max[0] + 1;
    Hmax = feature_max[1] + 1;
  }

  std::cout << "Loading " << number_of_frames << " Images [" << start_frame << ", "
            << start_frame + number_of_frames - 1 << "] .... " << std::flush;

  std::vector<cv::Mat> images = dataset.loadImageSequence(start_frame, number_of_frames);

  std::cout << "Done" << std::endl;

  const std::string windowname = "FeaturesXY: " + feature_name;
  cv::namedWindow(windowname);

  WaitKeyNavigation navigator(0, number_of_frames - 1, true);
  do {
    const int i = navigator.value();

    cv::Mat flow_field_img = cv::Mat::zeros(Hmax, Wmax, CV_8UC3);

#pragma omp parallel for
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        Eigen::Index p = i * W * H + y * W + x;
        int fx = std::round(featuresXY(0, p));
        int fy = std::round(featuresXY(1, p));

        flow_field_img.at<cv::Vec3b>(fy, fx) = images.at(i).at<cv::Vec3b>(y, x);
      }

    cv::imshow(windowname, flow_field_img);
    cv::displayOverlay(windowname, "Frame " + std::to_string(i + start_frame));
    cv::imwrite("Frame" + std::to_string(i + start_frame) + ".png", flow_field_img);

  } while (navigator());
}

}  // namespace vs

