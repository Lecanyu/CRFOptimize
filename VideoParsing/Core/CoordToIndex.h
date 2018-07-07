/**
 * @file CoordToIndex.h
 * @brief Pixel Coordinates To Index functors
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_CORE_COORD_TO_INDEX_H_
#define VIDEOPARSING_CORE_COORD_TO_INDEX_H_

#include <Eigen/Core>

namespace vp {

template<int D, typename T>
struct CoordToIndex;

/// Image pixel Coordinate To Index functor
template<typename T>
struct CoordToIndex<2, T> {
  CoordToIndex(T width, T height)
      : x_size(width) {
    max_index = x_size * height;
  }
  T operator()(T x, T y) const {
    return x + x_size * y;
  }

  template<class Derived>
  T operator()(const Eigen::MatrixBase<Derived>& coord) const {
    return coord.x() + x_size * coord.y();
  }

  T maxIndex() const {
    return max_index;
  }

 private:
  T x_size;
  T max_index;
};

/// Video pixel Coordinate To Index functor
template<typename T>
struct CoordToIndex<3, T> {
  CoordToIndex(T width, T height, T length)
      : x_size(width),
        xy_size(width * height),
        max_index(width * height * length) {
  }

  T operator()(T x, T y, T z) const {
    return x + x_size * y + xy_size * z;
  }

  template<class Derived>
  T operator()(const Eigen::MatrixBase<Derived>& coord) const {
    return coord.x() + x_size * coord.y() + xy_size * coord.z();
  }

  T maxIndex() const {
    return max_index;
  }

 private:
  T x_size;
  T xy_size;
  T max_index;
};

typedef CoordToIndex<2, int> CoordToIndex2i;
typedef CoordToIndex<2, unsigned> CoordToIndex2u;

typedef CoordToIndex<3, int> CoordToIndex3i;
typedef CoordToIndex<3, unsigned> CoordToIndex3u;

}  // namespace vp



#endif /* VIDEOPARSING_CORE_COORD_TO_INDEX_H_ */
