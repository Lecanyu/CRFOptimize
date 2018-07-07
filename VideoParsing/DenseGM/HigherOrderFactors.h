/**
 * @file HigherOrderFactors.h
 * @brief HigherOrderFactors
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_HIGHERORDERFACTORS_H_
#define VIDEOPARSING_DENSEGM_HIGHERORDERFACTORS_H_

#include <Eigen/Core>
#include <vector>

namespace vp {

/// ALE style segment potentials
class SegmentHOFactors {
 public:
  typedef std::vector<int> Segment;
  typedef std::vector<Segment> Segmentations;

  SegmentHOFactors(const Segmentations& segmentations,
                    const Eigen::MatrixXf& segment_probabilities,
                    const float param1 = 0.0025f,
                    const float param2 = 1.0f);


  Eigen::MatrixXf apply(const Eigen::MatrixXf& Q_in) const;

 private:
  Segmentations segments_;
  Eigen::MatrixXf segment_probabilities_;
  float ho_param1, ho_param2;
};

}  // namespace vp

#endif /* VIDEOPARSING_DENSEGM_HIGHERORDERFACTORS_H_ */
