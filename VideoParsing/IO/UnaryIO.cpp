/**
 * @file UnaryIO.cpp
 * @brief UnaryIO
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/UnaryIO.h"
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

namespace vp {

void loadUnaryFromUncompressedBinaryFile(const std::string& filepath, Eigen::MatrixXf& unary_data, int& width, int& height, int&bands) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open File at " + filepath);
  }

  file.read((char *) &width, 4);
  file.read((char *) &height, 4);
  file.read((char *) &bands, 4);

  unary_data.resize(bands, width * height);

  file.read((char *) unary_data.data(), sizeof(float) * unary_data.size());

  file.close();
}

void loadUnaryFromUncompressedBinaryFile(const std::string& filepath, cv::Mat& cost_image, int& width, int& height, int&bands) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open File at " + filepath);
  }

  file.read((char *) &width, 4);
  file.read((char *) &height, 4);
  file.read((char *) &bands, 4);

  cost_image.create(height, width, CV_32FC(bands));

  file.read((char *) cost_image.data, sizeof(float) * width * height * bands);
  file.close();
}

cv::Mat loadUnaryAsImage(const std::string& filepath) {
  namespace fs = boost::filesystem;
  fs::path fp(filepath);

  if (!fs::exists(fp)) {
    std::cout << "Provided unary_file path does not exist: " << fp << "\n";
    throw std::runtime_error("Unary File Not Found");
  }

  if (!fs::is_regular_file(fp)) {
    std::cout << "Provided unary_file is not a regular file: " << fp << "\n";
    throw std::runtime_error("Unary File is not a regular file");
  }

  int width, height, bands;
  cv::Mat unary_data;

  if(fp.extension() == ".bin")
    loadUnaryFromUncompressedBinaryFile(filepath, unary_data, width, height, bands);
  else
    throw std::runtime_error("Unknown Unary File Format");

  return unary_data;
}

Eigen::MatrixXf loadUnary(const std::string& filepath) {
  namespace fs = boost::filesystem;
  fs::path fp(filepath);

  if (!fs::exists(fp)) {
    std::cout << "Provided unary_file path does not exist: " << fp << "\n";
    throw std::runtime_error("Unary File Not Found");
  }

  if (!fs::is_regular_file(fp)) {
    std::cout << "Provided unary_file is not a regular file: " << fp << "\n";
    throw std::runtime_error("Unary File is not a regular file");
  }

  int width, height, bands;
  Eigen::MatrixXf unary_data;

  if(fp.extension() == ".bin")
    loadUnaryFromUncompressedBinaryFile(filepath, unary_data, width, height, bands);
  else
    throw std::runtime_error("Unknown Unary File Format");

  return unary_data;
}

Eigen::MatrixXf loadUnary(const std::string& filepath, int& width, int& height) {
  namespace fs = boost::filesystem;
  fs::path fp(filepath);

  if (!fs::exists(fp)) {
    std::cout << "Provided unary_file path does not exist: " << fp << "\n";
    throw std::runtime_error("Unary File Not Found");
  }

  if (!fs::is_regular_file(fp)) {
    std::cout << "Provided unary_file is not a regular file: " << fp << "\n";
    throw std::runtime_error("Unary File is not a regular file");
  }

  int bands;
  Eigen::MatrixXf unary_data;

  if(fp.extension() == ".bin")
    loadUnaryFromUncompressedBinaryFile(filepath, unary_data, width, height, bands);
  else
    throw std::runtime_error("Unknown Unary File Format");

  return unary_data;
}

void saveUnaryAsUncompressedBinaryFile(Eigen::MatrixXf& unary, const int W, const int H, const std::string& filepath) {

  if (unary.cols() != W * H) {
    throw std::runtime_error("unary.cols() != W*H");
  }

  int numberOfLabels = unary.rows();

  std::ofstream ofile(filepath, std::ios::binary);

  ofile.write((char*) &W, sizeof(W));
  ofile.write((char*) &H, sizeof(H));
  ofile.write((char*) &numberOfLabels, sizeof(numberOfLabels));
  ofile.write((char *) unary.data(), sizeof(float) * unary.size());

  ofile.close();
}

}  // namespace vp
