/**
 * @file PairwiseFunctions.h
 * @brief PairwiseFunctions
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_PAIRWISE_FUNCTIONS_H_
#define VIDEOPARSING_DENSEGM_PAIRWISE_FUNCTIONS_H_

#include <Eigen/Core>
#include <vector>


namespace vp {

/**
 * PottsFunction models a function \mu(a,b) = -w when a==b or 0 otherwise
 *  \mu(a,b) = -w[a==b]
 */
class PottsFunction {
 public:
  explicit PottsFunction(float potts_weight = 1.f)
      : weight(potts_weight) {
  }

  template<class Derived>
  typename Derived::PlainObject apply(const Eigen::MatrixBase<Derived>& Q) const {
    return -weight * Q;
  }

 private:
  float weight;
};

/** DiagonalCompatabilityFunction
 *
 *  \mu(a,b) = -[a==b]v(a)
 */
class DiagonalCompatabilityFunction {
 public:
  template <class Derived>
  DiagonalCompatabilityFunction(const Eigen::DenseBase<Derived>& weights_vec)
      : weights_(weights_vec) {
  }

  DiagonalCompatabilityFunction(const std::vector<float>& weights, bool negate = true)
      : weights_(Eigen::VectorXf(weights.size())) {
    for (Eigen::Index i = 0; i < weights_.size(); ++i) {
      weights_[i] = negate ? -weights[i] : weights[i];
    }
  }

  template<class Derived>
  typename Derived::PlainObject apply(const Eigen::MatrixBase<Derived>& Q) const {
    assert(Q.rows() == weights_.size());
    return weights_.asDiagonal() * Q;
  }
 private:
  Eigen::VectorXf weights_;
};

/// Implements matrix \mu(a,b) [enforces symmetry, but not positive definitness]
class MatrixCompatabilityFunction {
 public:
  template <class Derived>
  MatrixCompatabilityFunction(const Eigen::DenseBase<Derived>& weights_mat )
   : weights_(0.5*(weights_mat + weights_mat.transpose())) {
    assert(weights_.rows() == weights_.cols());
  }

  template<class Derived>
  typename Derived::PlainObject apply(const Eigen::MatrixBase<Derived>& Q) const {
    assert(Q.rows() == weights_.rows());
    return weights_ * Q;
  }

 private:
  Eigen::MatrixXf weights_;
};


}  // namespace vp

#endif /* VIDEOPARSING_DENSEGM_PAIRWISE_FUNCTIONS_H_ */
