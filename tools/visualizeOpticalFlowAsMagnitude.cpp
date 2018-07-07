/**
 * @file visualizeOpticalFlowAsMagnitude.cpp
 * @brief visualizeOpticalFlowAsMagnitude
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/OpticalFlowIO.h"
#include "VideoParsing/Utils/ColorizeFlow.h"
#include "VideoParsing/Utils/VisualizationHelper.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "ERROR: flo file path not provided\n";
    std::cout << "Usage: " << argv[0] << " /path/to/flo/files\n";
    return EXIT_FAILURE;
  }

  std::cout << "User provided " << argc - 1 << " input files\n";

  cv::namedWindow("Flow", 1);
  vp::WaitKeyNavigation navigator(1, argc - 1, true);
  do {
    const int i = navigator.value();
    cv::Mat flow = vp::readOpticalFlow(argv[i]);
    if (flow.empty()) {
      std::cout << "ERROR: Flow Field is empty\n";
      return EXIT_FAILURE;
    }
    cv::displayOverlay("Flow", "Flow Image #" + std::to_string(i));
    cv::imshow("Flow", vp::colorizeFlowWithMagnitude(flow));

  } while (navigator());


  return EXIT_SUCCESS;
}
