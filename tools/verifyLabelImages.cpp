/**
 * @file verifyLabelImages.cpp
 * @brief check Label Images if they have invalid labels colors
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Utils/SemanticLabel.h"

#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <iostream>


int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "ERROR: label image file paths not provided\n";
    std::cout << "Usage: " << argv[0] << "/path/to/label/images\n";
    return EXIT_FAILURE;
  }

  const int number_of_frames = argc - 1;
  std::cout << "User provided " << number_of_frames << " input files\n";

  int num_of_bad_images = 0;

  for (int i = 0; i < number_of_frames; ++i) {
    namespace bfs = boost::filesystem;

    bfs::path fp(argv[i+1]);

    if(!bfs::exists(fp)) {
      std::cout << "ERROR: " << fp << " does not exist\n";
      return EXIT_FAILURE;
    }

    if(!bfs::is_regular(fp)) {
      std::cout << "ERROR: " << fp << " is not a regular file\n";
      return EXIT_FAILURE;
    }

    cv::Mat label_img = cv::imread(fp.string());
    if (label_img.empty()) {
      std::cout << "ERROR: label_img is empty\n";
      return EXIT_FAILURE;
    }

    std::cout << "loaded Image " << fp.filename() << std::endl;

    bool bad_image = false;

    for (int y = 0; y < label_img.rows; ++y)
      for (int x = 0; x < label_img.cols; ++x) {
        cv::Vec3b color = label_img.at<cv::Vec3b>(y, x);
        try {
          vp::bgrToCamvid11(color);
        } catch (const std::invalid_argument& e) {
          std::cout << "File " << fp.filename() << " has Invalid color " << color << " at  pixel (" << x <<"," <<  y << ")\n";
          bad_image =  true;
          std::cout << std::flush;
        }
      }

    if(bad_image)
      ++num_of_bad_images;

  }

  std::cout << "Found " << num_of_bad_images << " bad images out of " << number_of_frames
            << " images checked\n";

  return EXIT_SUCCESS;
}



