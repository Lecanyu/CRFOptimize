/**
 * @file HigherOrderFactors.cpp
 * @brief HigherOrderFactors
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/DenseGM/HigherOrderFactors.h"

namespace vp {

SegmentHOFactors::SegmentHOFactors(const Segmentations& segmentations,
                                     const Eigen::MatrixXf& segment_probs,
                                     const float param1,
                                     const float param2)
    : segments_(segmentations),
      segment_probabilities_(segment_probs),
      ho_param1(param1),
      ho_param2(param2) {
}

Eigen::MatrixXf SegmentHOFactors::apply(const Eigen::MatrixXf& Q_in) const {
  const int L = Q_in.rows();  // number of labels
  const int N = Q_in.cols();  // number of variables

  Eigen::MatrixXf Q_out(L, N);
  Q_out.setZero();

  for (Segmentations::size_type segment_id = 0; segment_id < segments_.size(); ++segment_id) {
    const Segment& segment = segments_[segment_id];

    Eigen::VectorXf hnorm = Eigen::VectorXf::Ones(L);
    for (const int pixel_index : segment) {
      hnorm = hnorm.array() * Q_in.col(pixel_index).array();
    }

    float weight = 0.3f * segment.size();
    Eigen::VectorXf const_data = -weight * segment_probabilities_.col(segment_id).array().log();

    for (const int pixel_index : segment) {

      Eigen::VectorXf hnorm_by_Qi = hnorm.array() / (Q_in.col(pixel_index).array() + 0.0001f);

      Q_out.col(pixel_index) += ho_param1 * const_data - ho_param2 * hnorm_by_Qi;
    }
  }
  return Q_out;
}


}  // namespace vp

