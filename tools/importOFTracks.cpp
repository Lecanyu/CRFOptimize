/**
 * @file importOFTracks.cpp
 * @brief converts OFTracker tracks to our format (boost serialization)
 *
 * For example this can import the the output of Dense Point Tracking code of
 * http://lmb.informatik.uni-freiburg.de/resources/binaries/eccv2010_trackingLinux64.zip
 *
 * OFTracks format (See vp::readOFtracks() for more details)
 * Text file with following fields:
 * SequenceLength(int)
 * NumberOfTracks(int)
 * TrackLabel(int) TrackLength(int)
 * x1 y1 frame_num1
 * x2 y2 frame_num2
 * ..
 * ..
 * TrackLabel(int) TrackLength(int)
 * x1 y1 frame_num1
 * x2 y2 frame_num2
 * ..
 * ..
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/TracksIO.h"
#include <iostream>

int main(int argc, char **argv) {
  using std::cout;

  if (argc != 3) {
    cout << "ERROR: parsing inputs\n";
    cout << "Usage: " << argv[0] << " input_oftracks out_file\n";
    return EXIT_FAILURE;
  }

  std::vector<vp::Track2D> tracks = vp::readOFtracks(argv[1]);
  vp::saveTracks(tracks, argv[2]);
  cout << "Saved " << tracks.size() << " tracks as " << argv[2] << "\n";

  return EXIT_SUCCESS;
}



