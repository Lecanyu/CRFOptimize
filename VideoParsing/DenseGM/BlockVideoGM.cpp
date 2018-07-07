/**
 * @file BlockVideoGM.cpp
 * @brief BlockVideoGM
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/DenseGM/BlockVideoGM.h"
#include "VideoParsing/Core/EigenUtils.h"
#include <iostream>

namespace vp {

BlockVideoGM::BlockVideoGM(size_type W, size_type H, size_type T, size_type L)
  : W_(W), H_(H), L_(L){

  spatial_gms_.reserve(T);
  for(size_type i = 0; i< T; ++i)
    spatial_gms_.emplace_back(W_, H_, L_);
}

void BlockVideoGM::addUnaries(const std::vector<Eigen::MatrixXf>& unaries) {
  if (unaries.size() != spatial_gms_.size())
    throw std::length_error("No of unaries != # number of frames");

#pragma omp parallel for
  for (int i = 0; i < spatial_gms_.size(); ++i) {
    spatial_gms_[i].unary() = unaries[i];
  }
}

void BlockVideoGM::addPairwise3D(PairwiseFactor* pairwise_factor, size_type start_frame, size_type number_of_frames) {
  if((start_frame + number_of_frames) > spatial_gms_.size() )
    throw std::length_error("Adding pairwise Block outside max frame range");

  pairwise_ranges_.emplace_back(start_frame, number_of_frames);
  pairwise3D_.push_back(pairwise_factor);
}


std::vector<Eigen::VectorXi> BlockVideoGM::computeMAPestimate(size_type num_of_iterations) const {
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

Eigen::MatrixXf BlockVideoGM::runInference(size_type num_of_iterations) const {
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

    // Apply Block 3D pairwise factors
    for(size_type i= 0; i< pairwise3D_.size(); ++i) {
      Eigen::Index start_index = pairwise_ranges_[i].first * WH;
      Eigen::Index num_of_vars = pairwise_ranges_[i].second * WH;

      temps.block(0, start_index, L_, num_of_vars ) -= pairwise3D_[i].apply(Qmarginals.block(0, start_index, L_, num_of_vars));
    }

    // Apply higher order Potts factors
    for (const HOTrackFactor& track_factor : ho_track_factors_) {
      track_factor.apply(Qmarginals, temps);
    }

#pragma omp parallel for
    for(int i= 0; i< spatial_gms_.size(); ++i)
      Qmarginals.block(0, i * WH, L_, WH ) = expAndNormalize(temps.block(0, i * WH, L_, WH ));
  } // End of iterations

  return Qmarginals;

}

}  // end namespace vp
