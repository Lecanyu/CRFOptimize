/**
 * @file LinearSolverHelper.h
 * @brief LinearSolverHelper
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_FEATURES_LINEARSOLVERHELPER_H_
#define VIDEOPARSING_FEATURES_LINEARSOLVERHELPER_H_

#include "VideoParsing/Core/Track2D.h"
#include "VideoParsing/Core/CoordToIndex.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Sparse>

namespace vp {

template<class SparseMatrixType>
SparseMatrixType createTracksLaplacian(const std::vector<Track2D>& tracks,
                                            const int start_frame,
                                            const int number_of_frames,
                                            const int W, const int H) {

  typedef typename SparseMatrixType::Scalar Scalar;
  std::vector<Eigen::Triplet<Scalar>> coeffs;

  const CoordToIndex3i coordToindex(W, H, number_of_frames);

  for (const Track2D& track : tracks) {
    const int track_start_frame_id = std::max(start_frame, track.startFrameId());
    if (track_start_frame_id >= (start_frame + number_of_frames - 1))
      continue;

    const int track_end_frame_id = std::min(start_frame + number_of_frames, track.endFrameId());
    if (track_end_frame_id <= start_frame)
      continue;

    for(int frame_id = track_start_frame_id; frame_id < track_end_frame_id; ++frame_id) {
      const Eigen::Vector2i mloc = track.locationAtFrame(frame_id).array().round().cast<int>();
      const int mindex = coordToindex(mloc.x(), mloc.y(), frame_id - start_frame);

      std::vector<int> neighbor_indices;

      if(frame_id > track_start_frame_id) {
        const Eigen::Vector2i loc = track.locationAtFrame(frame_id - 1).array().round().cast<int>();
        neighbor_indices.push_back(coordToindex(loc.x(), loc.y(), frame_id - 1 - start_frame));
      }

      if(frame_id < (track_end_frame_id - 1)) {
        const Eigen::Vector2i loc = track.locationAtFrame(frame_id + 1).array().round().cast<int>();
        neighbor_indices.push_back(coordToindex(loc.x(), loc.y(), frame_id + 1 - start_frame));
      }

      const Scalar neigbor_wt = - Scalar(1) / neighbor_indices.size();
      for (int nindex : neighbor_indices) {
        coeffs.emplace_back(mindex, nindex, neigbor_wt);
      }
      coeffs.emplace_back(mindex, mindex, 1);
    }
  }

  SparseMatrixType laplacian(coordToindex.maxIndex(), coordToindex.maxIndex());
  laplacian.setFromTriplets(coeffs.begin(), coeffs.end());
  return laplacian;
}


template<class SparseMatrixType>
SparseMatrixType createGramMatrixFromTracks(const std::vector<Track2D>& tracks,
                                            const int start_frame, const int number_of_frames,
                                            const int W, const int H) {

  typedef typename SparseMatrixType::Scalar Scalar;
  std::vector<Eigen::Triplet<Scalar>> residual_coeffs;

  typedef CoordToIndex<3, Eigen::Index> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, number_of_frames);
  for (const Track2D& track : tracks) {
    const int track_start_frame_id = std::max(start_frame, track.startFrameId());
    if (track_start_frame_id >= (start_frame + number_of_frames - 1))
      continue;

    const int track_end_frame_id = std::min(start_frame + number_of_frames, track.endFrameId());
    if (track_end_frame_id <= start_frame)
      continue;

    for(int frame_id = track_start_frame_id; frame_id < (track_end_frame_id - 1); ++frame_id) {
      assert(track.isActive(frame_id));
      assert(track.isActive(frame_id + 1));

      const Eigen::Vector2i locA = track.locationAtFrame(frame_id).array().round().cast<int>();
      const Eigen::Vector2i locB = track.locationAtFrame(frame_id + 1).array().round().cast<int>();

      const Eigen::Index indexA = coordToindex(locA.x(), locA.y(), frame_id - start_frame);
      const Eigen::Index indexB = coordToindex(locB.x(), locB.y(), frame_id + 1 - start_frame);

      residual_coeffs.emplace_back(indexA, indexB, 1);
      residual_coeffs.emplace_back(indexA, indexA, -1);
    }
  }

  SparseMatrixType residuals(coordToindex.maxIndex(), coordToindex.maxIndex());
  residuals.setFromTriplets(residual_coeffs.begin(), residual_coeffs.end());
  return residuals.transpose() * residuals;
}

template<class SparseMatrixType>
SparseMatrixType createGramMatrixFromFlows(const std::vector<cv::Mat>& flows, bool forward = true) {
  typedef typename SparseMatrixType::Scalar Scalar;


  const int number_of_frames = flows.size() + 1;
  const int W = flows.front().cols;
  const int H = flows.front().rows;

  const CoordToIndex3i coordToindex(W, H, number_of_frames);
  std::vector<Eigen::Triplet<Scalar>> residual_coeffs;

  for (int i = 0; i < (number_of_frames - 1); ++i) {
    const cv::Mat& flow = flows[i];
    assert(W == flow.cols);
    assert(H == flow.rows);

    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {

        const cv::Vec2f uv = flow.at<cv::Vec2f>(y, x);
        int x_plus_u =  std::round(x + uv[0]);
        int y_plus_v =  std::round(y + uv[1]);

        if(x_plus_u < 0 || y_plus_v < 0 || x_plus_u >= W || y_plus_v >= H)
          continue;

        Eigen::Index indexA = forward ? coordToindex(x, y, i) : coordToindex(x_plus_u, y_plus_v, i);
        Eigen::Index indexB = forward ? coordToindex(x_plus_u, y_plus_v, i + 1) : coordToindex(x, y, i + 1);

        residual_coeffs.emplace_back(indexA, indexB, 1);
        residual_coeffs.emplace_back(indexA, indexA, -1);
      }
  }

  SparseMatrixType residuals(coordToindex.maxIndex(), coordToindex.maxIndex());
  residuals.setFromTriplets(residual_coeffs.begin(), residual_coeffs.end());
  return residuals.transpose() * residuals;
}

template<class SparseMatrixType>
SparseMatrixType createGramMatrixFromFlows(const std::vector<cv::Mat>& fwd_flows, const std::vector<cv::Mat>& bwd_flows, const float max_error = 5.0f) {
  typedef typename SparseMatrixType::Scalar Scalar;

  assert(fwd_flows.size() == bwd_flows.size());


  const int number_of_frames = fwd_flows.size() + 1;
  const int W = fwd_flows.front().cols;
  const int H = fwd_flows.front().rows;

  const CoordToIndex3i coordToindex(W, H, number_of_frames);
  std::vector<Eigen::Triplet<Scalar>> residual_coeffs;

  for (int i = 0; i < (number_of_frames - 1); ++i) {
    const cv::Mat& fwd_flow = fwd_flows.at(i);
    const cv::Mat& bwd_flow = bwd_flows.at(i);

    assert(W == fwd_flow.cols);
    assert(H == fwd_flow.rows);
    assert(W == bwd_flow.cols);
    assert(H == bwd_flow.rows);

    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {

        const cv::Vec2f fwd_uv = fwd_flow.at<cv::Vec2f>(y, x);
        int x_plus_u =  std::round(x + fwd_uv[0]);
        int y_plus_v =  std::round(y + fwd_uv[1]);

        if(x_plus_u < 0 || y_plus_v < 0 || x_plus_u >= W || y_plus_v >= H)
          continue;

        cv::Vec2f bwd_uv = bwd_flow.at<cv::Vec2f>(y_plus_v, x_plus_u);
        float badness = Eigen::Vector2f(x_plus_u + bwd_uv[0] - x, y_plus_v + bwd_uv[1] - y).norm();

        if(badness >  max_error)
          continue;

        Eigen::Index indexA = coordToindex(x, y, i);
        Eigen::Index indexB = coordToindex(x_plus_u, y_plus_v, i + 1);

        residual_coeffs.emplace_back(indexA, indexB, 1);
        residual_coeffs.emplace_back(indexA, indexA, -1);
      }
  }

  SparseMatrixType residuals(coordToindex.maxIndex(), coordToindex.maxIndex());
  residuals.setFromTriplets(residual_coeffs.begin(), residual_coeffs.end());
  return residuals.transpose() * residuals;
}

}  // namespace vp

#endif /* VIDEOPARSING_FEATURES_LINEARSOLVERHELPER_H_ */
