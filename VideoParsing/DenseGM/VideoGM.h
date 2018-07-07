/**
 * @file VideoGM.h
 * @brief VideoGM
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_VIDEOGM_H_
#define VIDEOPARSING_DENSEGM_VIDEOGM_H_

#include "VideoParsing/DenseGM/Dense2DGridGM.h"

namespace vp {

class VideoGM {
 public:
  typedef Dense2DGridGM::size_type size_type;
  typedef boost::ptr_vector<PairwiseFactor> PairwiseFactors;

  /**@brief Primary constructor
   *
   * @param W width of each frame
   * @param H height of each frame
   * @param T length (number of frames in video)
   * @param M number of labels
   */
  VideoGM(size_type W, size_type H, size_type T, size_type L);

  std::vector<Dense2DGridGM>& spatialGMs() {return spatial_gms_;}
  const std::vector<Dense2DGridGM>& spatialGMs() const {return spatial_gms_;}

  void addUnaries(const std::vector<Eigen::MatrixXf>& unaries);

  template<class CompatbilityFunction>
  void addSpatialPairwiseGaussian(float sx, float sy, const CompatbilityFunction& function,
                                  NormalizationType normalization_type = NORMALIZE_SYMMETRIC) {
#pragma omp parallel for
    for (int i = 0; i < spatial_gms_.size(); ++i) {
      spatial_gms_[i].addPairwiseGaussian(sx, sy, function, normalization_type);
    }

  }

  // Add a Bilateral pairwise potential with spatial standard deviations sx, sy and color standard deviations sr,sg,sb
  template<class CompatbilityFunction>
  void addSpatialPairwiseBilateral(float sx, float sy, float sr, float sg, float sb,
                                   const std::vector<cv::Mat>& color_images,
                                   const CompatbilityFunction& function,
                                   NormalizationType normalization_type = NORMALIZE_SYMMETRIC) {
    if(color_images.size() != spatial_gms_.size())
    {
		throw std::length_error("No of images != #of spatial graphical models");
		/*std::cout << "ERROR: No of images != #of spatial graphical models" << std::endl;
		exit(-1);*/
    }
      

  #pragma omp parallel for
    for(int i = 0; i< spatial_gms_.size(); ++i) {
      spatial_gms_[i].addPairwiseBilateral(sx, sy, sr, sg, sb, color_images[i], function, normalization_type);
    }
  }

  const PairwiseFactors& pairwise3D() const {return pairwise3D_;}
  PairwiseFactors& pairwise3D() {return pairwise3D_;}


  std::vector<Eigen::VectorXi> computeMAPestimate(size_type num_of_iterations) const;
  Eigen::MatrixXf runInference(size_type num_of_iterations) const;


  size_type width() const {return W_;}
  size_type height() const {return H_;}
  size_type numberOfFrames() const {return spatial_gms_.size();}
  size_type numberOfLabels() const {return L_;}


 private:
  std::vector<Dense2DGridGM> spatial_gms_;
  PairwiseFactors pairwise3D_;
  size_type W_, H_;
  size_type L_;
};

}  // end namespace vp

#endif /* VIDEOPARSING_DENSEGM_VIDEOGM_H_ */
