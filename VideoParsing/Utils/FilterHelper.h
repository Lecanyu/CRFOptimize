/**
 * @file FilterHelper.h
 * @brief FilterHelper
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_UTILS_FILTERHELPER_H_
#define VIDEOPARSING_UTILS_FILTERHELPER_H_

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <vector>

namespace vp {

Eigen::Matrix<double, 2, Eigen::Dynamic> createFeaturesXY(int width, int height, int number_of_frames);

Eigen::MatrixXf createFeaturesXY(float sx, float sy, int width, int height);

Eigen::MatrixXf createFeaturesXY(float sx, float sy, const cv::Mat& flow);

Eigen::Matrix<float, 5, Eigen::Dynamic> createFeaturesXYRGB(float sx, float sy, float sr, float sg,
                                                            float sb, const cv::Mat& color_image);

Eigen::Matrix<float, 7, Eigen::Dynamic> createFeaturesXYRGBUV(float sx, float sy, float sr, float sg,
                                                              float sb, float su, float sv,
                                                              const cv::Mat& color_image,
                                                              const cv::Mat& flow_image);

Eigen::MatrixXf createFeaturesXYRGB(float sx, float sy, float sr, float sg, float sb,
                                    const cv::Mat& color_image, const cv::Mat& flow);

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesRGB(float sr, float sg, float sb, const cv::Mat& color_image);

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesXYS(float sx, float sy, float std_dev, const cv::Mat& seg);

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesXYS(float sx, float sy, const cv::Mat& flow, float std_dev, const cv::Mat& seg);

Eigen::MatrixXf createFeaturesS(float std_dev, const cv::Mat& seg);

std::vector<Eigen::MatrixXf> createFeaturesXY(float sx, float sy, const std::vector<cv::Mat>& fwd_flows);

std::vector<Eigen::MatrixXf> createFeaturesXYRGB(
    float sx, float sy, float sr, float sg, float sb,
    const std::vector<cv::Mat>& flows,
    const std::vector<cv::Mat>& images);

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesXYT(float sx, float sy,
		float st, const std::vector<cv::Mat>& flows);

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesXYT(float sx, float sy, float st, int W, int H, int T);

Eigen::Matrix<float, 6, Eigen::Dynamic> createFeaturesXYTRGB(float sx, float sy, float st,
                                                             float sr, float sg, float sb,
                                                             const std::vector<cv::Mat>& flows,
                                                             const std::vector<cv::Mat>& images);

Eigen::Matrix<float, 6, Eigen::Dynamic> createFeaturesXYTRGB(float sx, float sy, float st,
                                                             float sr, float sg, float sb,
                                                             const std::vector<cv::Mat>& images);

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesRGB(
		float sr, float sg, float sb, const std::vector<cv::Mat>& images);


}  // namespace vp

#endif /* VIDEOPARSING_UTILS_FILTERHELPER_H_ */
