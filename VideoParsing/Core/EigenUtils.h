/**
 * @file EigenUtils.h
 * @brief EigenUtils
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_EIGENUTILS_H_
#define VIDEOPARSING_DENSEGM_EIGENUTILS_H_

#include <Eigen/Core>

namespace vp {

/**@brief Exponentiate and normalize a matrix i.e. softmax
 *
 * @param in
 * @return normalized exponential of in
 */
template <class Derived>
typename Derived::PlainObject expAndNormalize(const Eigen::MatrixBase<Derived>& in) {
  typename Derived::PlainObject out = in.rowwise() - in.colwise().maxCoeff();
  out = out.array().exp();
  out = out.array().rowwise() / out.colwise().sum().eval().array();
  return out;
}

/**@brief Compute column-wise max coefficient index
 *
 * @param in
 * @return array with index to the row position of the  column-wise max coefficients
 */
template <class Derived>
Eigen::VectorXi getColwiseMaxCoeffIndex(const Eigen::DenseBase<Derived>& in) {
  Eigen::VectorXi max_index(in.cols());
  for(typename Derived::Index i = 0; i< in.cols(); ++i) {
    in.col(i).maxCoeff(&max_index[i]);
  }
  return max_index;
}


/**@brief Compute column-wise min coefficient index
 *
 * @param in
 * @return array with index to the row position of the  column-wise min coefficients
 */
template <class Derived>
Eigen::VectorXi getColwiseMinCoeffIndex(const Eigen::DenseBase<Derived>& in) {
  Eigen::VectorXi max_index(in.cols());
  for(typename Derived::Index i = 0; i< in.cols(); ++i) {
    in.col(i).minCoeff(&max_index[i]);
  }
  return max_index;
}

/**@brief Compute row-wise max coefficient index
 *
 * @param in
 * @return array with index to the row position of the row-wise wise max coefficients
 */
template <class Derived>
Eigen::VectorXi getRowwiseMaxCoeffIndex(const Eigen::DenseBase<Derived>& in) {
  Eigen::VectorXi max_index(in.rows());
  for(typename Derived::Index i = 0; i< in.rows(); ++i) {
    in.row(i).maxCoeff(&max_index[i]);
  }
  return max_index;
}


/// add specific element along each columns
template<class DerivedMat, class DerivedVec>
typename DerivedMat::Scalar sumColwiseIndices(const Eigen::MatrixBase<DerivedMat>& matrix,
                                              const Eigen::MatrixBase<DerivedVec>& indices) {
  assert(matrix.cols() == indices.size());
  typename DerivedMat::Scalar sum = 0;
  for (typename DerivedMat::Index i = 0; i < matrix.cols(); ++i) {
    sum += matrix(indices[i], i);
  }
  return sum;
}

}  // namespace vp


#endif /* VIDEOPARSING_DENSEGM_EIGENUTILS_H_ */
