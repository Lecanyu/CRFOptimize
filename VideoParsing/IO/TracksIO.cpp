/**
 * @file TracksIO.cpp
 * @brief TracksIO
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/TracksIO.h"
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive & ar, vp::Track2D::PointType& point, const unsigned int version) {
    ar & make_nvp("x", point.x());
    ar & make_nvp("y", point.y());
}

template<class Archive>
void serialize(Archive & ar, vp::Track2D & track, const unsigned int version) {
  ar & make_nvp("start_frameid", track.start_frame_id);
  ar & make_nvp("locations", track.locations);
}

} // namespace serialization
} // namespace boost

namespace vp {

std::vector<Track2D> readTracks(const std::string& file_name) {
  namespace fs = boost::filesystem;

  fs::path track_fp(file_name);

  if (!fs::exists(track_fp)) {
    std::cout << "Provided track_file path does not exist: " << track_fp << "\n";
    throw std::runtime_error("Track File Not Found");
  }

  if (!fs::is_regular_file(track_fp)) {
    std::cout << "Provided track_file is not a regular file: " << track_fp << "\n";
    throw std::runtime_error("Track File is not a regular file");
  }

  std::vector<Track2D> tracks;

  if(track_fp.extension() == ".dat")
    tracks = readOFtracks(file_name);
  else
    tracks = readBoostSerializedTracks(file_name);

  return tracks;
}

std::vector<Track2D> readBoostSerializedTracks(const std::string & filename) {
  namespace fs = boost::filesystem;
  fs::path fp(filename);

  std::vector<Track2D> tracks;
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open " + filename);

  if (fp.extension().string() == ".xml") {
    boost::archive::xml_iarchive ia(ifs);
    ia >> BOOST_SERIALIZATION_NVP(tracks);
  }
  else if(fp.extension().string() == ".txt") {
    boost::archive::text_iarchive ia(ifs);
    ia >> BOOST_SERIALIZATION_NVP(tracks);
  }
  else {
    boost::archive::binary_iarchive ia(ifs);
    ia >> BOOST_SERIALIZATION_NVP(tracks);
  }
  ifs.close();

  return tracks;
}

std::vector<Track2D> readOFtracks(const std::string & file_name) {
  std::vector<Track2D> tracks;

  // Read FT File
  std::ifstream file(file_name.c_str());
  if(! file.is_open() ) {
    throw std::runtime_error("Cannot open File at " + file_name);
  }

  int sequence_length;
  file >> sequence_length;
  // Read number of tracks
  int tracks_count;
  file >> tracks_count;

  tracks.reserve(tracks_count);

  for (int i = 0; i < tracks_count; i++) {

    // Read label and length of track
    int label;
    int track_length;
    file >> label;
    file >> track_length;

    assert(track_length > 0);

    Track2D track;
    track.locations.resize(track_length);

    file >> track.locations[0].x();
    file >> track.locations[0].y();
    file >> track.start_frame_id;

    // Read x,y coordinates and frame number of the tracked point
    for (int j = 1; j < track_length; ++j) {
      file >> track.locations[j].x();
      file >> track.locations[j].y();
      int frame_id;
      file >> frame_id;
    }

//    if(track_length < 3)
//      continue;

    tracks.push_back(track);

  }
  file.close();

  return tracks;
}

void saveTracks(const std::vector<Track2D>& tracks,
                const std::string& filename) {
  namespace fs = boost::filesystem;

  fs::path fp(filename);
  std::ofstream ofs(filename.c_str());
  if (!ofs.is_open())
    throw std::runtime_error("Cannot save at " + filename);

  if(fp.extension().string() == ".xml") {
    boost::archive::xml_oarchive oa(ofs);
    oa << BOOST_SERIALIZATION_NVP(tracks);
  }
  else if(fp.extension().string() == ".txt") {
    boost::archive::text_oarchive oa(ofs);
    oa << BOOST_SERIALIZATION_NVP(tracks);
  }
  else {
    boost::archive::binary_oarchive oa(ofs);
    oa << BOOST_SERIALIZATION_NVP(tracks);
  }
  ofs.close();
}

void saveOFTracks(const std::vector<Track2D>& tracks,
                  const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open File at " + filename);
  }

  // Determine sequence length (Kinda ignored)
  int min_start_frame = std::numeric_limits<int>::max();
  int max_end_frame = std::numeric_limits<int>::lowest();

  for (const Track2D& track : tracks) {
    min_start_frame = std::min(min_start_frame, track.startFrameId());
    max_end_frame = std::max(max_end_frame, track.endFrameId());
  }

  int sequence_length = max_end_frame - min_start_frame;

  file << sequence_length << "\n";
  file << tracks.size() << "\n";

  for (const Track2D& track : tracks) {
    int track_label = 0;
    int track_length = track.locations.size() ;

    file << track_label << "\n";
    file << track_length << "\n";

    for (int i = 0 ; i < track_length; ++i) {
      file << track.locations[i].x() << " ";
      file << track.locations[i].y() << " ";
      file << (track.startFrameId() + i) << "\n";
    }
  }

  file.close();
}

}  // namespace vp
