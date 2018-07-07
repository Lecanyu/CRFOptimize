/**
 * @file VisualizeFeatures.h
 * @brief VisualizeFeatures
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_VISUALIZE_FEATURES_H_
#define VIDEOPARSING_VISUALIZE_FEATURES_H_

#include "VideoParsing/IO/Dataset.h"

namespace vp {

void visualizeFeaturesXY(const Eigen::Matrix<float, 2, Eigen::Dynamic>& featuresXY,
                         const Dataset& dataset, int start_frame, int number_of_frames,
                         const std::string& feature_name = "");

void visualizeFeaturesXYwShifting(Eigen::Matrix<float, 2, Eigen::Dynamic>& featuresXY,
                                  const Dataset& dataset, int start_frame, int number_of_frames,
                                  const std::string& feature_name = "");


}  // namespace vp

#endif /* VIDEOPARSING_VISUALIZE_FEATURES_H_ */
