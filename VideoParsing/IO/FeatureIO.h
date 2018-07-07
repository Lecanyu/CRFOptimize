/**
 * @file FeatureIO.h
 * @brief FeatureIO
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_IO_FEATUREIO_H_
#define VIDEOPARSING_IO_FEATUREIO_H_

#include "VideoParsing/IO/Dataset.h"
#include <Eigen/Core>

namespace vp {

struct FeatureInfo {
  int dimension;
  int width;
  int height;
  int number_of_frames;
  int scalar_size;
  int start_frame;
  std::string dataset_name;
};


void saveFeatures(const Eigen::MatrixXf& features,
                  const Dataset& dataset,
                  const int start_frame,
                  const std::string& feature_file);

void saveFeatures(const Eigen::MatrixXf& features, const FeatureInfo& info,
                  const std::string& feature_file);

std::pair<Eigen::MatrixXf, FeatureInfo> readFeatures(const std::string& feature_file);

}  // namespace vp

#endif /* VIDEOPARSING_IO_FEATUREIO_H_ */
