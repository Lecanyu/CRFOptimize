/**
 * @file mergeCamvid32ToCamvid11.cpp
 * @brief merge Camvid32 label images to Camvid11 images
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Utils/SemanticLabel.h"

#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <iostream>


int main(int argc, char **argv) {

  if (argc < 3) {
    std::cout << "ERROR: segmentation file paths not provided\n";
    std::cout << "Usage: " << argv[0] << "start_frame /path/to/label/images\n";
    return EXIT_FAILURE;
  }

  const int number_of_frames = argc - 1;
  std::cout << "User provided " << number_of_frames << " input files\n";

  for (int i = 0; i < number_of_frames; ++i) {
    namespace bfs = boost::filesystem;

    bfs::path fp(argv[i+1]);

    if(!bfs::exists(fp)) {
      std::cout << "ERROR: " << fp << " does not exist\n";
      return EXIT_FAILURE;
    }

    if(!bfs::is_regular(fp)) {
      std::cout << "ERROR: " << fp << " isnot a regular file\n";
      return EXIT_FAILURE;
    }

    cv::Mat camvid32_img = cv::imread(fp.string());
    if (camvid32_img.empty()) {
      std::cout << "ERROR: camvid32_img is empty\n";
      return EXIT_FAILURE;
    }

    std::cout << "loaded Image " << fp.filename() << std::endl;

    std::vector<vp::Camvid11> labels;
    labels.reserve(camvid32_img.rows * camvid32_img.cols);

    for (int y = 0; y < camvid32_img.rows; ++y)
      for (int x = 0; x < camvid32_img.cols; ++x) {
        cv::Vec3b color = camvid32_img.at<cv::Vec3b>(y, x);
        try {
          labels.push_back(mergeCamvid32ToCamvid11(vp::bgrToCamvid32(color)));
        } catch (const std::invalid_argument& e) {
          std::cout << "Invalid color " << color << " at  pixel (" << x <<"," <<  y << ")\n";
        }

      }

    cv::Mat label_img = vp::getRGBImageFromLabels(labels, camvid32_img.cols, camvid32_img.rows, 11);

    const std::string out_fp = fp.stem().string() + std::string(".png");
    std::cout << "Saved Image as " << out_fp << std::endl;
    cv::imwrite(out_fp, label_img);
  }

}



