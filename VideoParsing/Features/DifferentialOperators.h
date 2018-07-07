/**
 * @file DifferentialOperators.h
 * @brief DifferentialOperators
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_FEATURES_DIFFERENTIALOPERATORS_H_
#define VIDEOPARSING_FEATURES_DIFFERENTIALOPERATORS_H_

#include "VideoParsing/Core/CoordToIndex.h"
#include <Eigen/Sparse>

namespace vp {

template<class SparseMatrixType>
SparseMatrixType createFwdDiffMatX(const int W, const int H, const int T = 1) {
  typedef typename SparseMatrixType::Scalar Scalar;
  std::vector<Eigen::Triplet<Scalar>> fwd_diff_coeffs;

  typedef CoordToIndex<3, Eigen::Index> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, T);

  for (int t = 0; t < T; ++t) {
    for (int my = 0; my < H; ++my)
      for (int mx = 0; mx < W; ++mx) {
        Eigen::Index midx = coordToindex(mx, my, t);
        if ((mx + 1) < W) {
          fwd_diff_coeffs.emplace_back(midx, coordToindex(mx + 1, my, t), 1);
          fwd_diff_coeffs.emplace_back(midx, midx, -1);
        } else {
          fwd_diff_coeffs.emplace_back(midx, midx, 1);
          fwd_diff_coeffs.emplace_back(midx, coordToindex(mx - 1, my, t), -1);
        }
      }
  }
  SparseMatrixType D(coordToindex.maxIndex(), coordToindex.maxIndex());
  D.setFromTriplets(fwd_diff_coeffs.begin(), fwd_diff_coeffs.end());
  return D;
}

template<class SparseMatrixType>
SparseMatrixType createFwdDiffMatY(const int W, const int H, const int T = 1) {
  typedef typename SparseMatrixType::Scalar Scalar;
  std::vector<Eigen::Triplet<Scalar>> fwd_diff_coeffs;

  typedef CoordToIndex<3, Eigen::Index> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, T);

  for (int t = 0; t < T; ++t) {
    for (int my = 0; my < H; ++my)
      for (int mx = 0; mx < W; ++mx) {
        Eigen::Index midx = coordToindex(mx, my, t);
        if ((my + 1) < H) {
          fwd_diff_coeffs.emplace_back(midx, coordToindex(mx, my + 1, t), 1);
          fwd_diff_coeffs.emplace_back(midx, midx, -1);
        } else {
          fwd_diff_coeffs.emplace_back(midx, midx, 1);
          fwd_diff_coeffs.emplace_back(midx, coordToindex(mx, my - 1, t), -1);
        }
      }
  }
  SparseMatrixType D(coordToindex.maxIndex(), coordToindex.maxIndex());
  D.setFromTriplets(fwd_diff_coeffs.begin(), fwd_diff_coeffs.end());
  return D;
}

template<class SparseMatrixType>
SparseMatrixType createD2TD2(const int W, const int H, const int T = 1) {
  SparseMatrixType Dx = createFwdDiffMatX<SparseMatrixType>(W, H, T);
  SparseMatrixType Dy = createFwdDiffMatY<SparseMatrixType>(W, H, T);
  SparseMatrixType D2x = Dx * Dx;
  SparseMatrixType D2y = Dy * Dy;
  return D2x.transpose() * D2x + D2y.transpose() * D2y;
}

template<class SparseMatrixType>
SparseMatrixType createLaplacian(const int W, const int H, const int T = 1) {
  SparseMatrixType Dx = createFwdDiffMatX<SparseMatrixType>(W, H, T);
  SparseMatrixType Dy = createFwdDiffMatY<SparseMatrixType>(W, H, T);
  return Dx.transpose() * Dx + Dy.transpose() * Dy;
}

template<class SparseMatrixType, class Derived>
SparseMatrixType createD2TD2(const Eigen::EigenBase<Derived>& weights, const int W, const int H, const int T = 1) {
  SparseMatrixType Dx = createFwdDiffMatX<SparseMatrixType>(W, H, T);
  SparseMatrixType Dy = createFwdDiffMatY<SparseMatrixType>(W, H, T);
  SparseMatrixType D2x = Dx * Dx;
  SparseMatrixType D2y = Dy * Dy;
  return D2x.transpose() * weights.derived() * D2x + D2y.transpose() * weights.derived() * D2y;
}

template<class SparseMatrixType, class Derived>
SparseMatrixType createLaplacian(const Eigen::EigenBase<Derived>& weights, const int W, const int H, const int T = 1) {
  SparseMatrixType Dx = createFwdDiffMatX<SparseMatrixType>(W, H, T);
  SparseMatrixType Dy = createFwdDiffMatY<SparseMatrixType>(W, H, T);
  return Dx.transpose() * weights.derived() * Dx + Dy.transpose() * weights.derived() * Dy;
}

}  // namespace vp


#endif /* VIDEOPARSING_FEATURES_DIFFERENTIALOPERATORS_H_ */
