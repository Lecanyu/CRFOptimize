/**
 * @file Dense2DGridGM.h
 * @brief Dense2DGridGM
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_DENSE2DGRIDGM_H_
#define VIDEOPARSING_DENSEGM_DENSE2DGRIDGM_H_

#include "VideoParsing/DenseGM/DenseGM.h"
#include <opencv2/core/core.hpp>

namespace vp {

class Dense2DGridGM : public DenseGM {
 public:
  Dense2DGridGM(size_type W, size_type H, size_type M);
  Dense2DGridGM(size_type W, size_type H, const Eigen::MatrixXf& unary);

  // Add a Gaussian pairwise potential with standard deviation sx and sy
  template<class FunctionType, class KernelType = SingleThreadedDenseKernel2>
  void addPairwiseGaussian(float sx, float sy, const FunctionType& function,
                           NormalizationType normalization_type = NORMALIZE_SYMMETRIC) {
    Eigen::Matrix<float, 2, Eigen::Dynamic> feature(2, N_);
    for (size_type y = 0; y < H_; ++y)
      for (size_type x = 0; x < W_; ++x) {
        const size_type index = y * W_ + x;
        feature(0, index) = x / sx;
        feature(1, index) = y / sy;
      }
    typedef DensePairwiseFactor<KernelType, FunctionType> DensePairwisePottsFactor;
    pairwise_.push_back(new DensePairwisePottsFactor(feature, function, normalization_type));
  }

  void addPairwiseGaussian(float sx, float sy, const std::vector<float>& function_params,
                            NormalizationType normalization_type = NORMALIZE_SYMMETRIC);


  // Add a Bilateral pairwise potential with spatial standard deviations sx, sy and color standard deviations sr,sg,sb
  template<class FunctionType, class KernelType = SingleThreadedDenseKernel5>
  void addPairwiseBilateral(float sx, float sy, float sr, float sg, float sb,
                            const cv::Mat& color_image, const FunctionType& function,
                            NormalizationType normalization_type = NORMALIZE_SYMMETRIC) {

    Eigen::Matrix<float, 5, Eigen::Dynamic> feature(5, N_);
    for (size_type y = 0; y < H_; ++y)
      for (size_type x = 0; x < W_; ++x) {
        const size_type index = y * W_ + x;
        feature(0, index) = x / sx;
        feature(1, index) = y / sy;

        cv::Vec3b bgr = color_image.at<cv::Vec3b>(y, x);
        feature(2, index) = bgr[2] / sr;
        feature(3, index) = bgr[1] / sg;
        feature(4, index) = bgr[0] / sb;
      }

    typedef DensePairwiseFactor<KernelType, FunctionType> DensePairwisePottsFactor;
    pairwise_.push_back(new DensePairwisePottsFactor(feature, function, normalization_type));
  }

  void addPairwiseBilateral(float sx, float sy, float sr, float sg, float sb,
                            const cv::Mat& color_image, const std::vector<float>& function_params,
                            NormalizationType normalization_type = NORMALIZE_SYMMETRIC);

  size_type width() const {return W_;}
  size_type height() const {return H_;}

 protected:
  size_type W_, H_;
};

} // end namespace vp

#endif /* VIDEOPARSING_DENSEGM_DENSE2DGRIDGM_H_ */
