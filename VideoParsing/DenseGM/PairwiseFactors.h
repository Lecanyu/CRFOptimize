/**
 * @file PairwiseFactors.h
 * @brief PairwiseFactors
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_PAIRWISE_FACTORS_H_
#define VIDEOPARSING_DENSEGM_PAIRWISE_FACTORS_H_

#include "VideoParsing/DenseGM/PairwiseFunctions.h"
#include "VideoParsing/DenseGM/DenseKernels.h"

#include <iomanip>
#include <locale>

namespace vp {

namespace detail {

template<class T>
std::string formatWithCommas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}

}  // namespace detail


struct PairwiseFactor {
  virtual Eigen::MatrixXf apply(const Eigen::MatrixXf& Q) const = 0;
  virtual ~PairwiseFactor() {}
  virtual PairwiseFactor* clone() const = 0;
  virtual std::string printInfo() const = 0;
};

inline PairwiseFactor* new_clone(const PairwiseFactor& a) {
  return a.clone();
}

template<class KernelType, class FunctionType = PottsFunction>
class DensePairwiseFactor : public PairwiseFactor {
  typedef DensePairwiseFactor<KernelType, FunctionType> ThisType;

  KernelType kernel_;
  FunctionType func_;

 public:
  DensePairwiseFactor(const KernelType& kernel, const FunctionType& func)
      : kernel_(kernel),
        func_(func) {
  }


  template<class Derived>
  DensePairwiseFactor(const Eigen::MatrixBase<Derived>& features,
                       const FunctionType& func,
                       NormalizationType ntype = NORMALIZE_SYMMETRIC)
      : kernel_(features, ntype),
        func_(func) {
  }

  Eigen::MatrixXf apply(const Eigen::MatrixXf& Q) const {
    return func_.apply(kernel_.apply(Q));
  }

  ThisType* clone() const { return new ThisType(*this); }

  std::string printInfo() const {
    return "D = " + std::to_string(kernel_.filter().featureDimenssion())
        + "  N = " + detail::formatWithCommas(kernel_.filter().numOfFeatures())
        + "  M = " + detail::formatWithCommas(kernel_.filter().numOfLatticePoints());
  }
};


struct BipartitePairwiseFactor {
  virtual Eigen::MatrixXf applyA(const Eigen::MatrixXf& QA) const = 0;
  virtual Eigen::MatrixXf applyB(const Eigen::MatrixXf& QA) const = 0;
  virtual ~BipartitePairwiseFactor() {}
  virtual BipartitePairwiseFactor* clone() const = 0;
};

inline BipartitePairwiseFactor* new_clone(const BipartitePairwiseFactor& a) {
  return a.clone();
}


// PairwiseFactor for bipartite kernels
template<class KernelType, class FunctionType>
class DenseBipartitePairwiseFactor : public BipartitePairwiseFactor {
  typedef DenseBipartitePairwiseFactor<KernelType, FunctionType> ThisType;
  KernelType kernel_;
  FunctionType func_;
 public:
  DenseBipartitePairwiseFactor(const Eigen::MatrixXf& featuresA, const Eigen::MatrixXf& featuresB,
                               const FunctionType& func, NormalizationType ntype = NORMALIZE_SYMMETRIC)
      : kernel_(featuresA, featuresB, ntype),
        func_(func) {
  }

  Eigen::MatrixXf applyA(const Eigen::MatrixXf& QA) const {
    return func_.apply(kernel_.applyA(QA));
  }

  Eigen::MatrixXf applyB(const Eigen::MatrixXf& QB) const {
    return func_.apply(kernel_.applyB(QB));
  }

  ThisType* clone() const { return new ThisType(*this); }
};


}  // namespace vp


#endif /* VIDEOPARSING_DENSEGM_PAIRWISE_FACTORS_H_ */
