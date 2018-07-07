/**
 * @file FeatureIO.cpp
 * @brief FeatureIO
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/FeatureIO.h"
#include <fstream>

namespace vp {

void saveFeatures(const Eigen::MatrixXf& features, const Dataset& dataset,
                      const int start_frame,
                      const std::string& filepath) {

  FeatureInfo info;

  info.dimension = features.rows();
  info.width = dataset.imageWidth();
  info.height = dataset.imageHeight();
  info.number_of_frames = features.cols() / (dataset.imageWidth() * dataset.imageHeight());
  info.scalar_size = sizeof(float);
  info.start_frame = start_frame;
  info.dataset_name = dataset.name();

  saveFeatures(features, info, filepath);
}

void saveFeatures(const Eigen::MatrixXf& features, const FeatureInfo& info,
                      const std::string& filepath) {

  if (features.rows() != info.dimension)
    throw std::runtime_error("# of rows does not match with info!");

  if (features.cols() != (info.width * info.height * info.number_of_frames))
    throw std::runtime_error("# of cols does not match with info!");


  std::ofstream ofile(filepath, std::ios::out | std::ios::binary);

  int header_magic_number = 99997;
  ofile.write(reinterpret_cast<char*>(&header_magic_number), sizeof(int));


  ofile.write((char *)(&info.dimension), sizeof(int));
  ofile.write((char *)(&info.width), sizeof(int));
  ofile.write((char *)(&info.height), sizeof(int));
  ofile.write((char *)(&info.number_of_frames), sizeof(int));
  ofile.write((char *)(&info.scalar_size), sizeof(int));
  ofile.write((char *)(&info.start_frame), sizeof(int));

  int string_size = info.dataset_name.size();
  ofile.write(reinterpret_cast<char*>(&string_size), sizeof(int));
  ofile.write(info.dataset_name.c_str(), string_size);


  ofile.write((char *) features.data(), info.scalar_size * features.size());

  int footer_magic_number = 99998;
  ofile.write(reinterpret_cast<char*>(&footer_magic_number), sizeof(int));
  ofile.close();
}

std::pair<Eigen::MatrixXf, FeatureInfo> readFeatures(const std::string& filepath) {
  std::ifstream ifile(filepath, std::ios::in | std::ios::binary);
  if (!ifile.is_open()) {
    throw std::runtime_error("Cannot open File!");
  }

  int magic_number;
  ifile.read((char *)(&magic_number), sizeof(int));
  if(magic_number != 99997)
    throw std::runtime_error("Header magic number does not match for " + filepath);

  FeatureInfo info;

  ifile.read((char *)(&info.dimension), sizeof(int));
  ifile.read((char *)(&info.width), sizeof(int));
  ifile.read((char *)(&info.height), sizeof(int));
  ifile.read((char *)(&info.number_of_frames), sizeof(int));
  ifile.read((char *)(&info.scalar_size), sizeof(int));
  ifile.read((char *)(&info.start_frame), sizeof(int));
  {
    int string_size;
    ifile.read(reinterpret_cast<char*>(&string_size), sizeof(int));
    char* temp = new char[string_size+1];
    ifile.read(temp, string_size);
    temp[string_size] = '\0';
    info.dataset_name = temp;
    delete [] temp;
  }


  Eigen::MatrixXf features(info.dimension, info.width * info.height * info.number_of_frames);

  ifile.read((char *) features.data(), sizeof(float) * features.size());


  ifile.read((char *)(&magic_number), sizeof(int));
  if(magic_number != 99998)
    throw std::runtime_error("Footer magic number does not match for " + filepath);

  ifile.close();

  return std::make_pair(features, info);
}

}  // namespace vp
