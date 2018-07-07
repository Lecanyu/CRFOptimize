/**
 * @file Dense2DGridGM.cpp
 * @brief Dense2DGridGM
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/DenseGM/Dense2DGridGM.h"
#include "VideoParsing/Core/EigenUtils.h"
#include <cassert>

namespace vp {

Dense2DGridGM::Dense2DGridGM(size_type width, size_type height, size_type number_of_labels)
    : DenseGM(width * height, number_of_labels),
      W_(width), H_(height){
}

Dense2DGridGM::Dense2DGridGM(size_type width, size_type height, const Eigen::MatrixXf& unary)
    : DenseGM(unary),
      W_(width),
      H_(height) {
  if (numberOfVariables() != width * height)
    throw std::runtime_error("numberOfVariables != width * height");
}

void Dense2DGridGM::addPairwiseGaussian(float sx, float sy,
                                        const std::vector<float>& function_params,
                                        NormalizationType normalization_type) {
  if (function_params.size() == 1)
    addPairwiseGaussian(sx, sy, PottsFunction(function_params[0]), normalization_type);
  else if (function_params.size() == L_)
    addPairwiseGaussian(sx, sy, DiagonalCompatabilityFunction(function_params), normalization_type);
  else
    throw std::runtime_error("function_params has weird num of values");
}

void Dense2DGridGM::addPairwiseBilateral(float sx, float sy, float sr, float sg, float sb,
                                             const cv::Mat& color_image,
                                             const std::vector<float>& function_params,
                                             NormalizationType normalization_type) {
  if (function_params.size() == 1)
    addPairwiseBilateral(sx, sy, sr, sg, sb, color_image, PottsFunction(function_params[0]), normalization_type);
  else if (function_params.size() == L_)
    addPairwiseBilateral(sx, sy, sr, sg, sb, color_image, DiagonalCompatabilityFunction(function_params), normalization_type);
  else
    throw std::runtime_error("function_params has weird num of values");
}

}  // namespace vp
