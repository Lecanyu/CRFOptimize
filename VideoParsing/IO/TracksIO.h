/**
 * @file TracksIO.h
 * @brief TracksIO
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_IO_TRACKSIO_H_
#define VIDEOPARSING_IO_TRACKSIO_H_

#include "VideoParsing/Core/Track2D.h"

namespace vp {

/**@brief Read Tracks from boost serialized (e.g. binary *.tracks) or OFTracker(.dat) files
 *
 * Appropriate loader is decided based on file extension
 *
 * @param file_name
 * @return std::vector of Track2D
 */
std::vector<Track2D> readTracks(const std::string & file_name);

/**@brief Read Tracks via boost serialization (e.g. binary *.tracks)
 *
 * @param file_name
 * @return
 */
std::vector<Track2D> readBoostSerializedTracks(const std::string & file_name);

/**@brief Read Tracks from OFTracker(.dat) files
 *
 * This is normally much slower than a binary boost serialized format.
 * For big tracks use binary boost serialized format.
 *
 * @param file_name
 * @return std::vector of Track2D
 */
std::vector<Track2D> readOFtracks(const std::string & file_name);


/// saves Video tracks using Boost serialization
void saveTracks(const std::vector<Track2D>& tracks, const std::string& filename);


/// saves Video tracks using OFTracker format
void saveOFTracks(const std::vector<Track2D>& tracks, const std::string& filename);

}  // namespace vp



#endif /* VIDEOPARSING_IO_TRACKSIO_H_ */
