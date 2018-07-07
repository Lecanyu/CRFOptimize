/**
 * @file DenseKernels.h
 * @brief DenseKernels
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_DENSEKERNELS_H_
#define VIDEOPARSING_DENSEGM_DENSEKERNELS_H_

#include "VideoParsing/DenseGM/PermutohedralLattice.h"
#include "VideoParsing/DenseGM/PartialParallelPermutohedralLattice.h"
#include "VideoParsing/DenseGM/ConcurrentPermutohedralLattice.h"

namespace vp {

// The filter in the dense CRF can be normalized in a few different ways
enum NormalizationType {
  NO_NORMALIZATION,  // No normalization whatsoever (will lead to a substantial approximation error)
  NORMALIZE_BEFORE,    // Normalize before filtering (Not used, just there for completeness)
  NORMALIZE_AFTER,     // Normalize after filtering (original normalization in NIPS 11 work)
  NORMALIZE_SYMMETRIC,  // Normalize before and after (ICML 2013, low approximation error and preserves the symmetry of CRF)
};


template <int D, template<int> class Filter = PermutohedralLattice>
class StaticDenseKernel {

 public:
  typedef Filter<D> FilterType;
  typedef Eigen::Matrix<float, D, Eigen::Dynamic> FeatureMatrixType;

  template <class Derived>
  StaticDenseKernel(const Eigen::MatrixBase<Derived>& f, NormalizationType ntype)
      : ntype_(ntype),
        lattice_(f) {
    const Eigen::DenseIndex N = f.cols();

    norm_ = lattice_.compute(Eigen::RowVectorXf::Ones(N)).transpose();

    if (ntype_ == NO_NORMALIZATION) {
      float mean_norm = N / norm_.sum();
      norm_.setConstant(mean_norm);
    }
    else if (ntype_ == NORMALIZE_SYMMETRIC) {
      norm_ = (norm_.array() + 1e-20).sqrt().inverse();
    }
    else {
      norm_ = (norm_.array() + 1e-20).inverse();
    }
  }

  template<class Derived>
  typename Derived::PlainObject apply(const Eigen::MatrixBase<Derived>& Q) const {
    return filter(Q, false);
  }

  const FilterType& filter() const {return lattice_;}

 private:

  template<class Derived>
  typename Derived::PlainObject filter(const Eigen::MatrixBase<Derived>& Q, bool transpose) const {
    typename Derived::PlainObject out = Q;

    if (ntype_ == NORMALIZE_SYMMETRIC || (ntype_ == NORMALIZE_BEFORE && !transpose)
        || (ntype_ == NORMALIZE_AFTER && transpose))
      out *= norm_.asDiagonal();

    // Filter
    if (transpose)
      lattice_.compute(out, out, true);
    else
      lattice_.compute(out, out);

    // Normalize again
    if (ntype_ == NORMALIZE_SYMMETRIC || (ntype_ == NORMALIZE_BEFORE && transpose)
        || (ntype_ == NORMALIZE_AFTER && !transpose))
      out = out * norm_.asDiagonal();

    return out;
  }

 private:
  NormalizationType ntype_;
  FilterType lattice_;
  Eigen::VectorXf norm_;

};

typedef StaticDenseKernel<2, PermutohedralLattice> SingleThreadedDenseKernel2;
typedef StaticDenseKernel<3, PermutohedralLattice> SingleThreadedDenseKernel3;
typedef StaticDenseKernel<4, PermutohedralLattice> SingleThreadedDenseKernel4;
typedef StaticDenseKernel<5, PermutohedralLattice> SingleThreadedDenseKernel5;



}  // namespace vp

#endif /* VIDEOPARSING_DENSEGM_DENSEKERNELS_H_ */
