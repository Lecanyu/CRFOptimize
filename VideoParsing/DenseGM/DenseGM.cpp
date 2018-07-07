/**
 * @file DenseGM.cpp
 * @brief DenseGM
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/DenseGM/DenseGM.h"
#include "VideoParsing/Core/EigenUtils.h"
#include <cassert>

namespace vp {

DenseGM::DenseGM(size_type number_of_variables, size_type number_of_labels)
    : N_(number_of_variables),
      L_(number_of_labels),
      unary_(L_, N_) {
  unary_.setConstant(-std::log(1/L_));
}

DenseGM::DenseGM(const Eigen::MatrixXf& unary)
    : N_(unary.cols()),
      L_(unary.rows()),
      unary_(unary) {
}

Eigen::VectorXi DenseGM::computeMAPestimate(size_type num_of_iterations) const {
  // Run inference
  Eigen::MatrixXf Q = runInference(num_of_iterations);
  // Find the map
  return getColwiseMaxCoeffIndex(Q);
}

Eigen::MatrixXf DenseGM::runInference(size_type num_of_iterations) const {

  Eigen::MatrixXf Q = expAndNormalize(-unary_);

  for(size_type iter = 0; iter < num_of_iterations; ++iter ) {
    Q = expAndNormalize(applyFactors(Q));
  }
  return Q;
}

Eigen::MatrixXf DenseGM::applyFactors(const Eigen::MatrixXf& Qmarginal) const {

  // Apply unary
  Eigen::MatrixXf tmp1 = -unary_;

  // Apply pairwise factors
  for (const PairwiseFactor& pairwise_factors : pairwise_) {
    tmp1 -= pairwise_factors.apply(Qmarginal);
  }

  // Apply higher order factors
  for (const SegmentHOFactors& ho_factors : higher_order_) {
    tmp1 -= ho_factors.apply(Qmarginal);
  }

  return tmp1;
}

float DenseGM::evaluateEnergy(const Eigen::VectorXi& labels) const {
  return evaluateUnaryEnergy(labels) + evaluatePairwiseEnergy(labels);
}

float DenseGM::evaluateUnaryEnergy(const Eigen::VectorXi& labels) const {
  if(labels.size() != unary_.cols())
    throw std::runtime_error("Wrong number Of Variables");
  return sumColwiseIndices(unary_, labels);
}

float DenseGM::evaluatePairwiseEnergy(const Eigen::VectorXi& labels) const {
  if(labels.size() != N_) {
    throw std::runtime_error("Wrong number Of Variables");
  }
  float pairwise_energy = 0;

  // Build a fake current belief [hard assignment]
  Eigen::MatrixXf labelsQ(L_, N_);
  for (size_type i = 0; i < N_; ++i)
    for (size_type l = 0; l < L_; ++l)
      labelsQ(l, i) = (labels[i] == l);

  for (const PairwiseFactor& pairwise_factors : pairwise_) {
    Eigen::MatrixXf transformedQ = pairwise_factors.apply(labelsQ);

    for (size_type i = 0; i < N_; ++i)
      pairwise_energy += 0.5 * transformedQ(labels[i], i);

  }

  return pairwise_energy;
}

}  // namespace vp
