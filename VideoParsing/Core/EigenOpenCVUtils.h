/**
 * @file EigenOpenCVUtils.h
 * @brief EigenOpenCVUtils
 *
 * @author Abhijit Kundu
 */


#ifndef EIGEN_OPENCV_UTILS_H_
#define EIGEN_OPENCV_UTILS_H_

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

#include <boost/static_assert.hpp>

template<class Derived>
cv::Mat makeOpenCVImageForDisplay(const Eigen::DenseBase<Derived>& eigen_image) {
  cv::Mat image(eigen_image.rows(), eigen_image.cols(), CV_8UC1);

  typedef typename Derived::Scalar Scalar;

  const Scalar min_val = eigen_image.minCoeff();
  const Scalar max_val = eigen_image.maxCoeff();
  const Scalar ratio = 255.0 / (max_val - min_val);

  for (int i = 0; i < image.rows; ++i)
    for (int j = 0; j < image.cols; ++j)
      image.at<uchar>(i, j) = (eigen_image(i, j) - min_val) * ratio;

  return image;
}

template<class Derived>
cv::Mat makeOpenCVImageForDisplay(const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& tensor) {
  BOOST_STATIC_ASSERT_MSG(Derived::NumIndices <= 2, "Expects rank 1/2 tensors only");
  typedef typename Derived::Scalar Scalar;

  const Derived& tensor_derived =  static_cast<const Derived>(tensor);

  cv::Mat image(tensor_derived.dimension(0), tensor_derived.dimension(1), CV_8UC1);

  const Eigen::Tensor<Scalar, 1> min_val = tensor.minimum();
  const Eigen::Tensor<Scalar, 1> max_val = tensor.maximum();
  const Scalar ratio = 255.0 / (max_val(0) - min_val(0));

  for (int i = 0; i < image.rows; ++i)
    for (int j = 0; j < image.cols; ++j)
      image.at<uchar>(i, j) = (tensor_derived(i, j) - min_val(0)) * ratio;

  return image;
}

cv::Mat makeOpenCVImageForDisplay(const cv::Mat& cv_image) {
  if(cv_image.type() != CV_64FC1 || cv_image.type() != CV_32FC1)
    std::runtime_error("Not Supported");

  double min_val;
  double max_val;
  cv::minMaxLoc(cv_image, &min_val, &max_val);
  const double ratio = 255.0 / (max_val - min_val);

  cv::Mat image(cv_image.rows, cv_image.cols, CV_8UC1);
  for (int i = 0; i < image.rows; ++i)
    for (int j = 0; j < image.cols; ++j) {

      if(cv_image.type() == CV_64FC1)
        image.at<uchar>(i, j) = (cv_image.at<double>(i, j) - min_val) * ratio;
      else if(cv_image.type() == CV_32FC1)
        image.at<uchar>(i, j) = (cv_image.at<float>(i, j) - min_val) * ratio;
    }

  return image;
}

template <class Derived>
cv::Mat convertToImage8U(const Eigen::DenseBase<Derived>& x, const int W, const int H) {
  assert (x.rows() == W * H);

  cv::Mat img(H, W, CV_MAKETYPE(CV_8U, x.cols()));
  Eigen::Index p = 0;
  for(int i= 0; i < H; ++i)
    for(int j= 0; j < W; ++j) {

      switch (x.cols()) {
        case 1:
          img.at<uchar>(i, j) = x(p, 0);
          break;
        case 3:
          img.at<cv::Vec3b>(i, j) = cv::Vec3b(x(p, 0), x(p, 1), x(p, 2));
          break;
        default:
          throw std::runtime_error("Can only handle 3 or 1 channel image");
      }
      ++p;
    }
  return img;
}

template <class Derived>
cv::Mat convertToImage32F(const Eigen::DenseBase<Derived>& x, const int W, const int H) {
  assert (x.rows() == W * H);

  cv::Mat img(H, W, CV_MAKETYPE(CV_32F, x.cols()));
  Eigen::Index p = 0;
  for(int i= 0; i < H; ++i)
    for(int j= 0; j < W; ++j) {

      switch (x.cols()) {
        case 1:
          img.at<float>(i, j) = x(p, 0);
          break;
        case 2:
          img.at<cv::Vec2f>(i, j) = cv::Vec2f(x(p, 0), x(p, 1));
          break;
        case 3:
          img.at<cv::Vec3f>(i, j) = cv::Vec3f(x(p, 0), x(p, 1), x(p, 2));
          break;
        default:
          throw std::runtime_error("Can only handle 3, 2 or 1 channel image");
      }
      ++p;
    }
  return img;
}

template <class Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> extractImageChannel(const cv::Mat& image, const int channel = 0) {
  const int num_of_channels = image.channels();
  if(channel >= num_of_channels || channel < 0)
    throw std::runtime_error("UnHandled # of channels");

  const int W = image.cols;
  const int H = image.rows;
  const int WH = W * H;

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(WH);

  Eigen::Index index = 0;
  switch (num_of_channels) {
    case 1:
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          b[index++] = image.at<uchar>(y, x);
        }
      break;
    case 3:
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          b[index++] = image.at<cv::Vec3b>(y, x)[channel];
        }
      break;
    default:
      throw std::runtime_error("Image needs to be either 3 or 1 channel image");
  }
  return b;
}

template <class ReturnType>
ReturnType convertVec3b(const cv::Vec3b& cv_vec) {
  return ReturnType(cv_vec[0], cv_vec[1], cv_vec[2]);
}

#endif // end EIGEN_OPENCV_UTILS_H_
