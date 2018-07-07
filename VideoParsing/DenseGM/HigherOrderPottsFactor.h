/**
 * @file HigherOrderPottsFactor.h
 * @brief HigherOrderPottsFactor
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_HIGHER_ORDER_POTTS_FACTOR_H_
#define VIDEOPARSING_DENSEGM_HIGHER_ORDER_POTTS_FACTOR_H_

#include "VideoParsing/Core/EigenUtils.h"
#include <vector>

namespace vp {


// Standard Rigid P-2 potential
template <class Scalar=double>
class HigherOrderPottsFactor {
public:
  typedef std::vector<int> VariableIndices;

  HigherOrderPottsFactor(const VariableIndices& var_indices, float cost_all_same, float cost_all_not_same)
      : var_indices_(var_indices),
        cost_all_same_(cost_all_same),
        cost_all_not_same_(cost_all_not_same) {

  }

  template <class DerivedIn, class DerivedOut>
  void apply(const Eigen::MatrixBase<DerivedIn>& Q_in, Eigen::MatrixBase<DerivedOut> const & Qout) const {
    typedef typename Eigen::internal::plain_col_type<DerivedIn>::type ColVectorType;

    assert(Q_in.rows() == Qout.rows());
    const int L = Q_in.rows();  // number of labels

    ColVectorType log_hnorm = ColVectorType::Zero(L);

    for (const int var_index : var_indices_) {
      assert(var_index < Q_in.cols());
      log_hnorm.array() += (Q_in.col(var_index).array() + 0.00001).log();
    }

    ColVectorType hnorm = expAndNormalize(log_hnorm);

    for (const int var_index : var_indices_) {
      ColVectorType hnorm_by_Qi = hnorm.array() / (Q_in.col(var_index).array() + 0.00001);
      (const_cast< Eigen::MatrixBase<DerivedOut>& >(Qout)).col(var_index) -= (hnorm_by_Qi * cost_all_same_ + (1 - hnorm_by_Qi.array()).matrix() * cost_all_not_same_);
    }
  }

private:
  VariableIndices var_indices_;
  Scalar cost_all_same_;
  Scalar cost_all_not_same_;
};


}  // namespace vp


#endif /* VIDEOPARSING_DENSEGM_HIGHERORDERPOTTSFACTOR_H_ */
