/**
 * @file IndexToCoord.h
 * @brief Index to pixel coordinate functors
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_CORE_INDEX_TO_COORD_H_
#define VIDEOPARSING_CORE_INDEX_TO_COORD_H_

#include <tuple>
#include <cstdlib>

namespace vp {

template<int D, typename T>
struct IndexToCoord;

/// Functor to get image pixel coordinate from pixel index
template<typename T>
struct IndexToCoord<2, T> {
  IndexToCoord(T width, T height)
      : x_size(width) {
    max_index = x_size * height;
  }

  inline std::tuple<T, T> operator()(T index) const {
    std::div_t div_by_x = std::div(index, x_size);
    return std::tuple<T, T>(div_by_x.rem, div_by_x.quot);
  }

  inline T maxIndex() const {
    return max_index;
  }

 private:
  T x_size;
  T max_index;
};

/// Functor to get video pixel coordinate from pixel index
template<typename T>
struct IndexToCoord<3, T> {
  IndexToCoord(T width, T height, T length)
      : x_size(width),
        xy_size(width * height),
        max_index(width * height * length) {
  }


  inline std::tuple<T, T, T> operator()(T index) const {
    std::div_t div_by_xy = std::div(index, xy_size);
    std::div_t div_by_x = std::div(div_by_xy.rem, x_size);
    return std::tuple<T, T, T>(div_by_x.rem, div_by_x.quot, div_by_xy.quot);
  }

  T maxIndex() const {
    return max_index;
  }

 private:
  T x_size;
  T xy_size;
  T max_index;
};

typedef IndexToCoord<2, int> IndexToCoord2i;
typedef IndexToCoord<2, unsigned> IndexToCoord2u;

typedef IndexToCoord<3, int> IndexToCoord3i;
typedef IndexToCoord<3, unsigned> IndexToCoord3u;

}  // namespace vp



#endif /* VIDEOPARSING_CORE_INDEX_TO_COORD_H_ */
