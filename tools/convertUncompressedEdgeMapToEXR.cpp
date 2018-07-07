/**
 * @file convertUncompressedEdgeMapToEXR.cpp
 * @brief convertUncompressedEdgeMapToEXR
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/EdgesIO.h"
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main(int argc, char **argv) {
  const int number_of_unary_files = argc - 1;

  if (number_of_unary_files < 1) {
    std::cout << "ERROR: parsing inputs\n";
    std::cout << "Usage: " << argv[0] << " path_to_uncompressed_binary_edges_files\n";
    return EXIT_FAILURE;
  }

  for (int i = 0; i < number_of_unary_files; ++i) {
    namespace fs = boost::filesystem;
    fs::path fp(argv[i+1]);

    if (!fs::exists(fp)) {
      std::cout << "Provided edge_file path does not exist: " << fp << "\n";
      return EXIT_FAILURE;
    }

    if (!fs::is_regular_file(fp)) {
      std::cout << "Provided edge_file is not a regular file: " << fp << "\n";
      return EXIT_FAILURE;
    }

    cv::Mat edge_image = vp::readEdgeProbAsImage(fp.string());

    fs::path out_fp = fp.filename().replace_extension(".exr");

    std::cout << "Saving as " << out_fp << "\n";
    cv::imwrite(out_fp.string(), edge_image);
  }

  return EXIT_SUCCESS;
}



