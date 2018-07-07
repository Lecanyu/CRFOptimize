/**
 * @file EdgesIO.cpp
 * @brief EdgesIO
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/EdgesIO.h"
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>

namespace vp {

cv::Mat readEdgeProbAsImageFromUncompressedBinary(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open File: " + filepath);
  }

  int rows, cols, channels, scalar_size;
  file.read((char *) &rows, 4);
  file.read((char *) &cols, 4);
  file.read((char *) &channels, 4);
  file.read((char *) &scalar_size, 4);

  if (scalar_size != sizeof(float)) {
    throw std::runtime_error("Edge file needs to be a 32 floating point image!");
  }

  cv::Mat edge_prob_image(rows, cols, CV_32FC(channels));
  file.read((char *) edge_prob_image.data, sizeof(float) * rows * cols * channels);

  file.close();

  return edge_prob_image;
}

Eigen::VectorXf readEdgeProbFromUncompressedBinary(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open File: " + filepath);
  }

  int rows, cols, channels, scalar_size;
  file.read((char *) &rows, 4);
  file.read((char *) &cols, 4);
  file.read((char *) &channels, 4);
  file.read((char *) &scalar_size, 4);

  if (scalar_size != sizeof(float)) {
    throw std::runtime_error("This expects a 32 floating point image!");
  }

  if (channels != 1) {
    throw std::runtime_error("This expects a single channel image!");
  }

  Eigen::VectorXf edge_data(rows * cols);

  file.read((char *) edge_data.data(), sizeof(float) * edge_data.size());

  file.close();

  return edge_data;
}


cv::Mat readEdgeProbAsImageFromEXR(const std::string& filepath) {
  cv::Mat edge_map =  cv::imread(filepath, cv::IMREAD_UNCHANGED);
  if (edge_map.empty())
    throw std::runtime_error("Edge File is empty");

  if (edge_map.depth() != CV_32F) {
    throw std::runtime_error("Edge map expects a 32 floating point image!");
  }

  if (edge_map.channels() != 1) {
    throw std::runtime_error("Edge map expects a single channel image!");
  }

  return edge_map;
}

Eigen::VectorXf readEdgeProbFromEXR(const std::string& filepath) {
  cv::Mat edge_image =  cv::imread(filepath, cv::IMREAD_UNCHANGED);
  if (edge_image.empty())
    throw std::runtime_error("Edge File is empty");

  if (edge_image.depth() != CV_32F) {
    throw std::runtime_error("Edge map expects a 32 floating point image!");
  }

  if (edge_image.channels() != 1) {
    throw std::runtime_error("Edge map expects a single channel image!");
  }

  Eigen::VectorXf edge_data(edge_image.rows * edge_image.cols);
  for(int i = 0; i< edge_image.rows; ++i)
    for(int j = 0; j< edge_image.cols; ++j) {
      edge_data[i*edge_image.cols + j] = edge_image.at<float>(i, j);
    }

  return edge_data;
}


cv::Mat readEdgeProbAsImage(const std::string& filepath) {
  namespace fs = boost::filesystem;
  fs::path fp(filepath);

  if (!fs::exists(fp)) {
    std::cout << "Provided edge_file path does not exist: " << fp << "\n";
    throw std::runtime_error("Edge File Not Found");
  }

  if (!fs::is_regular_file(fp)) {
    std::cout << "Provided edge_file is not a regular file: " << fp << "\n";
    throw std::runtime_error("Edge File is not a regular file");
  }

  if(fp.extension() == ".sed" || fp.extension() == ".bin")
    return readEdgeProbAsImageFromUncompressedBinary(filepath);
  else if(fp.extension() == ".exr")
    return readEdgeProbAsImageFromEXR(filepath);
  else
    throw std::runtime_error("Unknown Edge File Format");
}

Eigen::VectorXf readEdgeProb(const std::string& filepath) {
  namespace fs = boost::filesystem;
  fs::path fp(filepath);

  if (!fs::exists(fp)) {
    std::cout << "Provided edge_file path does not exist: " << fp << "\n";
    throw std::runtime_error("Edge File Not Found");
  }

  if (!fs::is_regular_file(fp)) {
    std::cout << "Provided edge_file is not a regular file: " << fp << "\n";
    throw std::runtime_error("Edge File is not a regular file");
  }

  if(fp.extension() == ".sed" || fp.extension() == ".bin")
    return readEdgeProbFromUncompressedBinary(filepath);
  else if(fp.extension() == ".exr")
      return readEdgeProbFromEXR(filepath);
  else
    throw std::runtime_error("Unknown Edge File Format");
}

}  // namespace vp


