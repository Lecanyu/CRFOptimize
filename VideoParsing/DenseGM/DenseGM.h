/**
 * @file DenseGM.h
 * @brief DenseGM
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_DENSEGM_H_
#define VIDEOPARSING_DENSEGM_DENSEGM_H_

#include "VideoParsing/DenseGM/PairwiseFactors.h"
#include "VideoParsing/DenseGM/HigherOrderFactors.h"
#include <boost/ptr_container/ptr_vector.hpp>

namespace vp {

class DenseGM {
 public:
  typedef std::size_t size_type;
  typedef boost::ptr_vector<PairwiseFactor> PairwiseFactors;

  DenseGM(size_type number_of_variables, size_type number_of_labels );
  DenseGM(const Eigen::MatrixXf& unary );


  size_type numberOfLabels() const {return L_;}
  size_type numberOfVariables() const {return N_;}

  const Eigen::MatrixXf& unary() const {return unary_;}
  Eigen::MatrixXf& unary() {return unary_;}

  const PairwiseFactors& pairwise() const {return pairwise_;}
  PairwiseFactors& pairwise() {return pairwise_;}

  const std::vector<SegmentHOFactors>& higherOrderFactors() const {return higher_order_;}
  std::vector<SegmentHOFactors>& higherOrderFactors() {return higher_order_;}

  /**@brief compute the MAP estimate
   *
   * @param num_of_iterations
   * @return a vector of MAP labels of size number_of_variables x 1
   */
  Eigen::VectorXi computeMAPestimate(size_type num_of_iterations) const;

  /**@brief runs the inference and return the Q matrix
   *
   * @param num_of_iterations
   * @return probabilities with dimension number_of_labels X number_of_variables
   */
  Eigen::MatrixXf runInference(size_type num_of_iterations) const;


  Eigen::MatrixXf applyFactors(const Eigen::MatrixXf& Qmarginal) const;

  /// Evaluate the energy (un-normalized) for a particular label assignment
  /// TODO Add HO energy
  float evaluateEnergy(const Eigen::VectorXi& labels) const;

  // Evaluate the unary energy for a particular label assignment
  float evaluateUnaryEnergy(const Eigen::VectorXi& labels) const;

  // Evaluate the pairwise energy for a particular label assignment
  float evaluatePairwiseEnergy(const Eigen::VectorXi& labels) const;


 protected:

  // number of  variables
  size_type N_;

  // number of labels
  size_type L_;

  // Unary Matrix NumOflabels X NumOfVariables
  Eigen::MatrixXf unary_;

  PairwiseFactors pairwise_;

  std::vector<SegmentHOFactors> higher_order_;
};

} // end namespace vp

#endif /* VIDEOPARSING_DENSEGM_DENSEGM_H_ */
