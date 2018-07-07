/**
 * @file Track2D.h
 * @brief Track2D
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_CORE_TRACK2D_H_
#define VIDEOPARSING_CORE_TRACK2D_H_

#include <Eigen/Core>
#include <vector>

namespace vp {

struct Track2D {
  typedef Eigen::Vector2f PointType;

  int start_frame_id;
  std::vector<PointType> locations;

  int startFrameId() const {return start_frame_id;}
  int endFrameId() const {return start_frame_id + locations.size();}

  bool isActive(int frame_id) const {
    return (frame_id >= start_frame_id) && (frame_id < (start_frame_id + (int)locations.size()));
  }

  const PointType& locationAtFrame(int frame_id) const {
    return locations[frame_id - start_frame_id];
  }
};

/// Functor To scale track locations
struct ScaleTrack2D {
  ScaleTrack2D(float scale) : scale_(scale) {};

  void operator()(Track2D& track) const {
    for(Track2D::PointType& point : track.locations)
      point *= scale_;
  }

private:
  float scale_;
};

/// Unary predicate useful for sorting Tracks based on length
struct LongerTrack {
  bool operator()(const Track2D& a, const Track2D& b) const {
    return a.locations.size() > b.locations.size();
  }
};

/// Unary predicate to check if a track can be active in some frame-id range
struct IsValidTrack {
  IsValidTrack(int start_frame, int number_of_frames)
      : start_frame_(start_frame),
        end_frame_(start_frame + number_of_frames) {
  }

  bool operator()(const Track2D& track) const {
    int track_start_frame = std::max(track.startFrameId(), start_frame_);
    int track_end_frame = std::min(track.endFrameId(), end_frame_);

    return (track_start_frame < (end_frame_ - 1)) && (track_end_frame > (start_frame_ + 1));
  }

 private:
  int start_frame_;
  int end_frame_;
};

/// Unary predicate to check if a track can be active in some frame-id range
struct IsActiveTrack {
  IsActiveTrack(int frame_id)
      : frame_id_(frame_id){
  }

  bool operator()(const Track2D& track) const {
    return track.isActive(frame_id_);
  }

 private:
  int frame_id_;
};


}  // namespace vp



#endif /* VIDEOPARSING_CORE_TRACK2D_H_ */
