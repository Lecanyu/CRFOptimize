/**
 * @file SemanticLabel.h
 * @brief SemanticLabel
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_UTILS_SEMANTICLABEL_H_
#define VIDEOPARSING_UTILS_SEMANTICLABEL_H_

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <cassert>

namespace vp {

/// Camvid 11 class Label Space
enum class Camvid11 {
  VOID = -1,       //!< VOID
  BUILDING = 0,    //!< BUILDING
  TREE = 1,        //!< TREE
  SKY = 2,         //!< SKY
  CAR = 3,         //!< CAR
  SIGN_SYMBOL = 4, //!< SIGN_SYMBOL
  ROAD = 5,        //!< ROAD
  PEDESTRIAN = 6,  //!< PEDESTRIAN
  FENCE = 7,       //!< FENCE
  COLUMN_POLE = 8, //!< COLUMN_POLE
  SIDEWALK = 9,    //!< SIDEWALK
  BICYCLIST = 10,  //!< BICYCLIST
};

/// Camvid 32 class labels
enum class Camvid32 {
  ANIMAL,
  ARCHWAY,
  BICYCLIST,
  BRIDGE,
  BUILDING,
  CAR,
  CARTLUGGAGEPRAM,
  CHILD,
  COLUMN_POLE,
  FENCE,
  LANEMKGSDRIV,
  LANEMKGSNONDRIV,
  MISC_TEXT,
  MOTORCYCLESCOOTER,
  OTHERMOVING,
  PARKINGBLOCK,
  PEDESTRIAN,
  ROAD,
  ROADSHOULDER,
  SIDEWALK,
  SIGNSYMBOL,
  SKY,
  SUVPICKUPTRUCK,
  TRAFFICCONE,
  TRAFFICLIGHT,
  TRAIN,
  TREE,
  TRUCK_BUS,
  TUNNEL,
  VEGETATIONMISC,
  VOID,
  WALL
};

/// Cityscape Label space
enum class CityScape19 {
  IGNORE = -1,
  ROAD = 0,
  SIDEWALK = 1,
  BUILDING = 2,
  WALL = 3,
  FENCE = 4,
  POLE = 5,
  TRAFFIC_LIGHT = 6,
  TRAFFIC_SIGN = 7,
  VEGETATION = 8,
  TERRAIN = 9,
  SKY = 10,
  PERSON = 11,
  RIDER = 12,
  CAR = 13,
  TRUCK = 14,
  BUS = 15,
  TRAIN = 16,
  MOTORCYCLE = 17,
  BICYCLE = 18,
};


/// Get Camvid32 label from bgr
Camvid32 bgrToCamvid32(const cv::Vec3b& bgr);

/// Get Camvid11 label from bgr
Camvid11 bgrToCamvid11(const cv::Vec3b& bgr);

/// Get CityScape19 label from bgr
CityScape19 bgrToCityScape19(const cv::Vec3b& bgr);

/// merge Camvid32 To Camvid11 labels
Camvid11 mergeCamvid32ToCamvid11(Camvid32 label);

/// get BGR value from Camvid11
template <class ReturnType =cv::Vec3b>
ReturnType labelToBGR(Camvid11 label) {
  ReturnType bgr;
  switch (label) {
    case Camvid11::BUILDING: // Building
      bgr = ReturnType(0, 0, 128);
      break;
    case Camvid11::TREE:  // Tree
      bgr = ReturnType(0, 128, 128);
      break;
    case Camvid11::SKY:  // Sky
      bgr = ReturnType(128, 128, 128);
      break;
    case Camvid11::CAR:  // car
      bgr = ReturnType(128, 0, 64);
      break;
    case Camvid11::SIGN_SYMBOL: // SignSymbol
      bgr = ReturnType(128, 128, 192);
      break;
    case Camvid11::ROAD: // Road
      bgr = ReturnType(128, 64, 128);
      break;
    case Camvid11::PEDESTRIAN: // Pedestrian
      bgr = ReturnType(0, 64, 64);
      break;
    case Camvid11::FENCE: // Fence
      bgr = ReturnType(128, 64, 64);
      break;
    case Camvid11::COLUMN_POLE: // Column Pole
      bgr = ReturnType(128, 192, 192);
      break;
    case Camvid11::SIDEWALK: // Sidewalk
      bgr = ReturnType(192, 0, 0);
      break;
    case Camvid11::BICYCLIST: // BiCyclist
      bgr = ReturnType(192, 128, 0);
      break;
    default: // Void
      bgr = ReturnType(0, 0, 0);
  }
  return bgr;
}

/// get BGR value from Camvid11
template <class ReturnType =cv::Vec3b>
ReturnType labelToBGR(CityScape19 label) {
  ReturnType bgr;
  switch (label) {
    case CityScape19::ROAD:
      bgr = ReturnType(128, 64,128);
      break;
    case CityScape19::SIDEWALK:
      bgr = ReturnType(232, 35, 244);
      break;
    case CityScape19::BUILDING:
      bgr = ReturnType(70, 70, 70);
      break;
    case CityScape19::WALL:
      bgr = ReturnType(156, 102, 102);
      break;
    case CityScape19::FENCE:
      bgr = ReturnType(153, 153, 190);
      break;
    case CityScape19::POLE:
      bgr = ReturnType(153, 153, 153);
      break;
    case CityScape19::TRAFFIC_LIGHT:
      bgr = ReturnType(30, 170, 250);
      break;
    case CityScape19::TRAFFIC_SIGN:
      bgr = ReturnType(0, 220, 220);
      break;
    case CityScape19::VEGETATION:
      bgr = ReturnType(35, 142, 107);
      break;
    case CityScape19::TERRAIN:
      bgr = ReturnType(152,251,152);
      break;
    case CityScape19::SKY:
      bgr = ReturnType(180,130,70);
      break;
    case CityScape19::PERSON:
      bgr = ReturnType(60, 20, 220);
      break;
    case CityScape19::RIDER:
      bgr = ReturnType(0, 0, 255);
      break;
    case CityScape19::CAR:
      bgr = ReturnType(142, 0, 0);
      break;
    case CityScape19::TRUCK:
      bgr = ReturnType(70, 0, 0);
      break;
    case CityScape19::BUS:
      bgr = ReturnType(100, 60, 0);
      break;
    case CityScape19::TRAIN:
      bgr = ReturnType(100, 80, 0);
      break;
    case CityScape19::MOTORCYCLE:
      bgr = ReturnType(230, 0, 0);
      break;
    case CityScape19::BICYCLE:
      bgr = ReturnType(32, 11, 119);
      break;
    default: // Void
      bgr = ReturnType(0, 0, 0);
  }
  return bgr;
}

/**@brief Get colored (RGB) image from labels
 *
 * Currently simply uses the number of labels to determine the colormap
 *
 * @param labeling
 * @param width
 * @param height
 * @param num_of_labels
 * @return colored (RGB) images
 */
template<class Derived>
cv::Mat getRGBImageFromLabels(const Eigen::MatrixBase<Derived>& labeling,
                              const int width, const int height,
                              const int num_of_labels) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY (Derived);

  cv::Mat map_labels(height, width, CV_8UC3);
  typename Derived::Index pixel_index = 0;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      switch (num_of_labels) {
        case 11:
          map_labels.at<cv::Vec3b>(y, x) = labelToBGR(
              static_cast<Camvid11>(labeling[pixel_index++]));
          break;
        case 19:
          map_labels.at<cv::Vec3b>(y, x) = labelToBGR(
              static_cast<CityScape19>(labeling[pixel_index++]));
          break;
      }
    }
  return map_labels;
}

/**@brief Get colored (RGB) image from labels with additional valid mask
 *
 * Currently simply uses the number of labels to determine the colormap
 *
 * @param labeling
 * @param validity
 * @param width
 * @param height
 * @param num_of_labels
 * @return colored (RGB) images
 */
template<class DerivedL, class DerivedV>
cv::Mat getRGBImageFromLabels(const Eigen::MatrixBase<DerivedL>& labeling,
                              const Eigen::MatrixBase<DerivedV>& validity,
                              const int width, const int height,
                              const int num_of_labels) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY (DerivedL);
  assert(labeling.size() == validity.size());
  cv::Mat map_labels(height, width, CV_8UC3);
  typename DerivedL::Index pixel_index = 0;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      if (validity[pixel_index])
        switch (num_of_labels) {
          case 11:
            map_labels.at<cv::Vec3b>(y, x) = labelToBGR(
                static_cast<Camvid11>(labeling[pixel_index++]));
            break;
          case 19:
            map_labels.at<cv::Vec3b>(y, x) = labelToBGR(
                static_cast<CityScape19>(labeling[pixel_index++]));
            break;
        }
      else
        map_labels.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
      ++pixel_index;
    }
  return map_labels;
}

/**@brief Get colored (RGB) image from labels with additional valid mask
 *
 * Currently simply uses the number of labels to determine the colormap
 *
 * @param labeling std::vector of label indices
 * @param validity
 * @param width
 * @param height
 * @param num_of_labels
 * @return colored (RGB) images
 */
template <class T, class A>
cv::Mat getRGBImageFromLabels(const std::vector<T, A>& labeling, const int width, const int height, const int num_of_labels) {
  cv::Mat map_labels(height, width, CV_8UC3);
  typename std::vector<T, A>::size_type pixel_index = 0;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      switch (num_of_labels) {
        case 11:
          map_labels.at<cv::Vec3b>(y, x) = labelToBGR(
              static_cast<Camvid11>(labeling[pixel_index++]));
          break;
        case 19:
          map_labels.at<cv::Vec3b>(y, x) = labelToBGR(
              static_cast<CityScape19>(labeling[pixel_index++]));
          break;
      }
    }
  return map_labels;
}

} // end namespace vp

#endif /* VIDEOPARSING_UTILS_SEMANTICLABEL_H_ */
