/**
 * @file FeatureHelper.h
 * @brief FeatureHelper
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_FEATURES_FEATUREHELPER_H_
#define VIDEOPARSING_FEATURES_FEATUREHELPER_H_

#include "VideoParsing/Core/CoordToIndex.h"
#include <opencv2/core/core.hpp>
#include <vector>


namespace vp {

template<class Scalar = double>
Eigen::Matrix<Scalar, Eigen::Dynamic, 2> computeFeaturesXY(const int W, const int H, const int number_of_frames = 1) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> featuresXY(W * H * number_of_frames, 2);

  typedef CoordToIndex<3, Eigen::Index> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, number_of_frames);

  for(int t = 0; t< number_of_frames; ++t) {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        Eigen::Index index = coordToindex(x,y,t);
        featuresXY(index, 0) = x;
        featuresXY(index, 1) = y;
      }
  }
  return featuresXY;
}


template<class Scalar = double>
Eigen::Matrix<Scalar, Eigen::Dynamic, 2> computeFeaturesXY(const std::vector<cv::Mat>& fwd_flows,
                                                           const std::vector<cv::Mat>& bwd_flows,
                                                           const int fix_frame) {
  const int T = fwd_flows.size() + 1;
  const int W = fwd_flows.front().cols;
  const int H = fwd_flows.front().rows;

  if (fix_frame < 0 || fix_frame >= T) {
    throw std::runtime_error("Fix frame outside valid range");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> featuresXY(W * H * T, 2);

  typedef CoordToIndex<3, Eigen::Index> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, T);


  {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        Eigen::Index index = coordToindex(x,y,fix_frame);
        featuresXY(index, 0) = x;
        featuresXY(index, 1) = y;
      }
  }

  for(int t = (fix_frame -1); t >=0; --t) {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        cv::Vec2f uv = fwd_flows.at(t).at<cv::Vec2f>(y, x);

        Eigen::Index indexA = coordToindex(x,y,t);
        Eigen::Index indexB = coordToindex(x,y,t+1);
        featuresXY(indexA, 0) = featuresXY(indexB, 0) + uv[0];
        featuresXY(indexA, 1) = featuresXY(indexB, 1) + uv[1];
      }
  }

  for (int t = fix_frame + 1; t < T; ++t) {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        cv::Vec2f uv = fwd_flows.at(t-1).at<cv::Vec2f>(y, x);

        Eigen::Index indexA = coordToindex(x, y, t - 1);
        Eigen::Index indexB = coordToindex(x, y, t);
        featuresXY(indexB, 0) = featuresXY(indexA, 0) - uv[0];
        featuresXY(indexB, 1) = featuresXY(indexA, 1) - uv[1];
      }

  }

  return featuresXY;
}


template<class Scalar = double>
Eigen::Matrix<Scalar, Eigen::Dynamic, 2> computeFeaturesXY(const std::vector<cv::Mat>& fwd_flows) {
  const int T = fwd_flows.size() + 1;
  const int W = fwd_flows.front().cols;
  const int H = fwd_flows.front().rows;

  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> featuresXY(W * H * T, 2);

  typedef CoordToIndex<3, Eigen::Index> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, T);


  {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        Eigen::Index index = coordToindex(x,y,T-1);
        featuresXY(index, 0) = x;
        featuresXY(index, 1) = y;
      }
  }

  for (int t = (T - 2); t >= 0; --t) {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        cv::Vec2f uv = fwd_flows.at(t).at<cv::Vec2f>(y, x);

        Eigen::Index indexA = coordToindex(x, y, t);
        Eigen::Index indexB = coordToindex(x, y, t + 1);
        featuresXY(indexA, 0) = featuresXY(indexB, 0) + uv[0];
        featuresXY(indexA, 1) = featuresXY(indexB, 1) + uv[1];
      }
  }

  return featuresXY;
}

}  // namespace vp

#endif /* VIDEOPARSING_FEATURES_FEATUREHELPER_H_ */
