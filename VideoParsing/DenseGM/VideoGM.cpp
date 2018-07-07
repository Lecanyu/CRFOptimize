/**
 * @file VideoGM.cpp
 * @brief VideoGM
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/DenseGM/VideoGM.h"
#include "VideoParsing/Core/EigenUtils.h"
#include <iostream>

namespace vp {

VideoGM::VideoGM(size_type W, size_type H, size_type T, size_type L)
  : W_(W), H_(H), L_(L){

  spatial_gms_.reserve(T);
  for(size_type i = 0; i< T; ++i)
    spatial_gms_.emplace_back(W_, H_, L_);
}

void VideoGM::addUnaries(const std::vector<Eigen::MatrixXf>& unaries) {
  if (unaries.size() != spatial_gms_.size())
    throw std::length_error("No of unaries != # number of frames");

#pragma omp parallel for
  for (int i = 0; i < spatial_gms_.size(); ++i) {
    spatial_gms_[i].unary() = unaries[i];
  }
}

std::vector<Eigen::VectorXi> VideoGM::computeMAPestimate(size_type num_of_iterations) const {
  // Run inference
  Eigen::MatrixXf Q = runInference(num_of_iterations);

  size_type WH = W_ * H_;

  std::vector<Eigen::VectorXi> map_estimates(spatial_gms_.size());

#pragma omp parallel for
  for (int t = 0; t < spatial_gms_.size(); ++t) {
    map_estimates[t] = getColwiseMaxCoeffIndex(
        Q.block(0, t * WH, L_, WH));
  }

  return map_estimates;
}

Eigen::MatrixXf VideoGM::runInference(size_type num_of_iterations) const {
  size_type WH = W_ * H_;

  Eigen::MatrixXf Qmarginals(L_, WH * spatial_gms_.size());

#pragma omp parallel for
  for(int i= 0; i< spatial_gms_.size(); ++i) {
    Qmarginals.block(0, i * WH, L_, WH ) = expAndNormalize(-spatial_gms_[i].unary());
  }

  // Do this for each iteration
  for(size_type iter = 0; iter < num_of_iterations; ++iter ) {
    std::cout << "Iteration # " << iter << std::endl;

    Eigen::MatrixXf temps(L_, WH * spatial_gms_.size());

#pragma omp parallel for
    for(int i= 0; i< spatial_gms_.size(); ++i) {

      //Apply 2D Spatial factors
      temps.block(0, i * WH, L_, WH ) = spatial_gms_[i].applyFactors(Qmarginals.block(0, i * WH, L_, WH ));
    }

    // Apply 3D pairwise factors
    for (const PairwiseFactor& pairwise_factors : pairwise3D_) {
      temps -= pairwise_factors.apply(Qmarginals);
    }

#pragma omp parallel for
    for(int i= 0; i< spatial_gms_.size(); ++i)
      Qmarginals.block(0, i * WH, L_, WH ) = expAndNormalize(temps.block(0, i * WH, L_, WH ));
  } // End of iterations

  return Qmarginals;

}


}  // end namespace vp
